#%%
from __future__ import annotations

import itertools
import multiprocessing as mp
import operator
import os
import pickle
from collections import Counter, deque
from typing import Generator, Literal, Optional, Union

import gymnasium as gym
import joblib
import mo_gymnasium as mo_gym
import networkx as nx
import numpy as np
import scipy
import simpy
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance
from numpy.typing import ArrayLike
from scipy.sparse.csgraph import construct_dist_matrix, shortest_path
from statsmodels.stats.weightstats import DescrStatsW

import numba_helpers
from helpers import (_get_coordinates_from_locations,
                     calculate_expected_pick_time, create_warehouse_grid,
                     sample_quantities, sample_volume_weights,
                     sample_volume_weights_with_categories)

np.seterr(invalid="ignore")

class Warehouse(gym.Env):
    """Gym environment for a collaborative picking warehouse."""        
    def __init__(self, x: int, y: int, nr_pickers: int, nr_amrs: int,
                 bipartite_state: bool = False, two_sided_aisles=True,
                 has_congestion: bool = True, use_warehouse_coordinates: bool = True,
                 amr_speed: float =1.5, # 1.5
                 amr_speed_std: float = 0.15,
                 picker_speed: float = 1.25, # 1.25
                 picker_speed_std: float = 0.15,
                 overtake_penalty: float = 15,
                 overtake_penalty_std: float = 2.5,
                 picking_time: float = 7.5,
                 picking_time_std: float = 0.5,
                 allow_picker_waiting: bool = True, 
                 allow_pick_encounter_vehicles: bool = False,
                 dict_type_state_reward_info: bool = True,
                 max_steps: int = 100,
                 generate_pickruns_in_advance: bool = True,
                 milp_compatible: bool = False,
                 only_current_destinations_available: bool = True,
                 ensure_diverse_initial_pickruns: bool = True,
                 pickrun_len_minimum: int = 15,
                 pickrun_len_maximum: int = 25, 
                 fixed_seed: Optional[int] = None,
                 benchmark_warehouse: bool = False,
                 disruption_freq: Optional[int] = None,
                 disruption_time: float = 60,
                 disruption_time_std: float = 7.5,
                 use_randomness: bool = False,
                 use_real_pick_times: bool = False,
                 locations_per_category: bool = True,
                 feat_category: Literal["both", "effic", "fair"] = "both") -> None:
        """Instantiates a new warehouse environment.
        """        
        super(Warehouse, self).__init__()
        self.feat_category = feat_category
        self.bipartite_state = bipartite_state
        self.sim = simpy.Environment() # Initialize a simpy simulation
        self.fixed_seed = fixed_seed
        self.benchmark_warehouse = benchmark_warehouse
        self.disruption_freq = disruption_freq
        self.disruption_time = disruption_time
        self.disruption_time_std = disruption_time_std
        self.use_randomness = use_randomness
        self.locations_per_category = locations_per_category
        if not hasattr(self, "reward_history"):
            self.reward_history = {"time": [], "fairness": [], "workload_values": []}
        
        self.pick_counter = {idx: 0 for idx in range(nr_pickers)}
        # Store the dimensions of the warehouse
        self.x = x  # Number of aisles
        self.y = y  # Number of racks per aisle (per side)
        self.two_sided_aisles = two_sided_aisles
        self.has_congestion = has_congestion
        self.use_warehouse_coordinates = use_warehouse_coordinates
        self.pickrun_len_minimum = pickrun_len_minimum
        self.pickrun_len_maximum = pickrun_len_maximum
        if use_warehouse_coordinates:
            self.coordinates_y, self.coordinates_x = _get_coordinates_from_locations(
                np.arange(x*y*2), y, x,True
            )
        self.allow_picker_waiting = allow_picker_waiting
        self.allow_picking_encountered_vehicles = allow_pick_encounter_vehicles
        self.dict_type_state_reward_info = dict_type_state_reward_info
        self.allow_pick_encounter_vehicles = allow_pick_encounter_vehicles # NOT YET USED
        self.generate_pickruns_in_advance = generate_pickruns_in_advance
        self.only_current_destinations_available = only_current_destinations_available
        self.use_real_pick_times = use_real_pick_times
        # Store time parameters
        self.amr_speed = amr_speed
        self.amr_speed_std = amr_speed_std
        self.picker_speed = picker_speed
        self.picker_speed_std = picker_speed_std
        self.overtake_penalty = overtake_penalty
        self.overtake_penalty_std = overtake_penalty_std
        self.picking_time = picking_time
        self.picking_time_std = picking_time_std
        self.max_steps = max_steps
        self.steps = 0
        self.milp_compatible = milp_compatible
        self.ensure_diverse_initial_pickruns = ensure_diverse_initial_pickruns
        if milp_compatible:
            self.max_steps_end = self.max_steps
        self.current_fairness = 0
        
        # Create a an adjacency matrix and distance matrix for the warehouse
        self.adjacency_matrix_uni = self._create_adjacency_matrix(
            is_unidirectional=True
        )
        self.distance_matrix_uni = self._create_distance_matrix(
            is_unidirectional=True
        ) 
        
        self.adjacency_matrix_bi = self._create_adjacency_matrix(
            is_unidirectional=False
        )
        self.distance_matrix_bi = self._create_distance_matrix(
            is_unidirectional=False
        )
        
        self.adjacency_list_bi = self._create_adjacency_list(
            is_unidirectional=False
        )
        self.adjacency_list_uni = self._create_adjacency_list(
            is_unidirectional=True
        )
        # edge_links is used for the state generation,
        # as coo is preferred format by pytorch geometric
        self.edge_links = self.adjacency_matrix_bi.tocoo()
        
        # Initialize the action space and observation space, which are needed in gym
        if self.two_sided_aisles:
            self.action_space = spaces.Discrete(int(x * y * 2))
        else:
            self.action_space = spaces.Discrete(int(x * y))
        if self.feat_category == "both":
            self.num_node_features = 36
        elif self.feat_category == "effic":
            self.num_node_features = 23 # 23
        elif self.feat_category == "fair":
            self.num_node_features = 13
        else:
            raise ValueError("feat_category must be one of 'both', 'effic' or 'fair'")
        self.observation_space = spaces.Dict(
            {"graph": spaces.Graph(
                node_space=spaces.Box(low=0, high=50000, shape=(
                    self.num_node_features,), dtype=np.float64),
                edge_space=spaces.Discrete(1)
            ),
                "mask": spaces.Sequence(spaces.Discrete(2)),
                "Aisle_nrs": spaces.Sequence(spaces.Discrete(self.x)),
                "picks_left": spaces.Discrete(self.max_steps)})
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        # Create entities with initial locations
        sample_locations_amrs = list(self.np_random.choice(range(0, self.action_space.n), nr_amrs))
        if self.generate_pickruns_in_advance:
            self.pickruns = self.pregenerate_pickruns(nr_amrs)
            initial_destinations_amrs = [pickrun[0][0] for pickrun in self.pickruns]
            for amr_id in range(nr_amrs):
                if sample_locations_amrs[amr_id] == initial_destinations_amrs[amr_id]:
                    sample_locations_amrs[amr_id] = self.np_random.choice(
                        [location for location in range(0, self.action_space.n)
                         if (location != sample_locations_amrs[amr_id])]
                    )
        self.amrs = [AMR(self, location, id) for id, location
                     in enumerate(sample_locations_amrs)]
        # Picker start locations should not be an AMR location or destination
        sample_locations_pickers = list(self.np_random.choice(
            [location for location in range(0, self.action_space.n)
             if location not in [amr.destination for amr in self.amrs]
             + [amr.location for amr in self.amrs]],
            nr_pickers
        ))
        self.pickers = [Picker(self, location, id) for id, location
                        in enumerate(sample_locations_pickers)]
        self._initial_assignment_picker_destination()
        self.entities = self.pickers + self.amrs # All entities in the warehouse
        # Create the racks, which manage the picking interaction between pickers and AMRs
        self.racks = self._create_racks()
        if not self.locations_per_category:
            try:
                self.volume_weight_info = sample_volume_weights(
                    self.action_space.n, generator=self.np_random
                )
            except ValueError:
                self.volume_weight_info = sample_volume_weights(
                    self.action_space.n, generator=self.np_random, with_replacement=True
                )
        else:
            self.volume_weight_info = sample_volume_weights_with_categories(
                warehouse=self, nr_samples=self.action_space.n, generator=self.np_random
            )
            
        # Create an initial observation
        self.observation = self._create_observation(self.pickers[0]) #TDDO: Check when multiple pickers
        # Create the mask, which is used to determine which actions are available
        self.mask = self.get_mask()
        
        self.state_dict = {}
        
    def _initial_assignment_picker_destination(self) -> None:
        initial_amr_destinations = [amr.destination for amr in self.amrs]
        temp_distance_matrix = self.distance_matrix_bi.copy()
        assigned = []
        for picker in self.pickers:
            picker_assigned = False
            counter = 0
            while (not picker_assigned) and (counter < len(initial_amr_destinations)):
                id_of_amr_with_closest_destination = int(np.argmin(
                    temp_distance_matrix[picker.location, initial_amr_destinations]
                ))
                if id_of_amr_with_closest_destination not in assigned:
                    self._assign_location_to_picker(
                        picker,
                        self.amrs[id_of_amr_with_closest_destination].destination
                    )
                    assigned += [id_of_amr_with_closest_destination]
                    picker_assigned = True
                else:
                    temp_distance_matrix[
                        picker.location,
                        self.amrs[id_of_amr_with_closest_destination].destination] = np.inf
                counter += 1
            if not picker_assigned:
                id_of_amr_with_closest_next_destination = int(np.argmin(
                    temp_distance_matrix[picker.location, 
                                         [amr.pickrun[0][0] for amr in self.amrs
                                          if amr.pickrun[0][0] not in [
                                              other_picker.destination
                                              for other_picker in self.pickers
                                              ]
                                          ]
                                         ]))
                self._assign_location_to_picker(
                    picker,
                    self.amrs[id_of_amr_with_closest_next_destination].pickrun[0][0]
                )
                
                    
    def reset(self, seed=None, options=None) -> tuple[Union[GraphInstance, dict[int, GraphInstance]], dict]:
        """Reset the environment to a random initial state for a new episode.

        Returns:
            GraphInstance: An initial state representation.
        """
        if not self.fixed_seed is None:
            seed = self.fixed_seed
        self._reset_with_seed(seed=seed)
        if self.milp_compatible:
            self.max_steps_end = self.max_steps - len([item for pickrun in self.pickruns for item in pickrun])
            self.pickruns = deque([])
        for entity in self.entities:
            entity.reset_simpy_processes()
         
        state_dict = {}
        continue_running = True
        while continue_running:
            anyof_event = simpy.events.AnyOf(
                self.sim,
                [picker.path_finished_event for picker in self.pickers]
            )
            self.sim.run(until=anyof_event)
            triggered_picker_id = int(np.where([picker.path_finished_event.triggered
                                                for picker in self.pickers])[0])
            triggered_picker = self.pickers[triggered_picker_id]
            if not (self.racks[triggered_picker.location].items != [] and
                ("AMR" in list(zip(*self.racks[triggered_picker.location].items))[0])):
                 continue_running = False
            else:
                for entity in self.entities:
                    entity.reset_simpy_processes()

        mask = self.get_mask()
        info = {"mask": mask, "previous_mask": self.mask}
        if self.dict_type_state_reward_info:
            for picker in self.pickers:
                state_dict[picker.id] = self._create_observation(picker)
            return state_dict, info
        else:
            self.active_picker_id = int(
                np.where([picker.path_finished_event.triggered
                          for picker in self.pickers])[0]
            )
            state = self._create_observation(self.pickers[self.active_picker_id])
            return state, info
        
    def _reset_with_seed(self, seed: Optional[int]) -> None:
        super().reset(seed=seed)
        self.__init__(self.x, self.y, len(self.pickers), len(self.amrs),
                      bipartite_state=self.bipartite_state,
                      two_sided_aisles=self.two_sided_aisles,
                      has_congestion=self.has_congestion,
                      use_warehouse_coordinates=self.use_warehouse_coordinates,
                      amr_speed=self.amr_speed, 
                      amr_speed_std=self.amr_speed_std,
                      picker_speed=self.picker_speed,
                      picker_speed_std=self.picker_speed_std,
                      overtake_penalty=self.overtake_penalty,
                      overtake_penalty_std=self.overtake_penalty_std,
                      picking_time=self.picking_time,
                      picking_time_std=self.picking_time_std,
                      allow_picker_waiting=self.allow_picker_waiting,
                      allow_pick_encounter_vehicles=self.allow_pick_encounter_vehicles,
                      dict_type_state_reward_info=self.dict_type_state_reward_info,
                      max_steps=self.max_steps,
                      generate_pickruns_in_advance=self.generate_pickruns_in_advance,
                      milp_compatible=self.milp_compatible,
                      only_current_destinations_available=self.only_current_destinations_available,
                      ensure_diverse_initial_pickruns=self.ensure_diverse_initial_pickruns,
                      pickrun_len_maximum=self.pickrun_len_maximum,
                      pickrun_len_minimum=self.pickrun_len_minimum,
                      fixed_seed=self.fixed_seed,
                      benchmark_warehouse=self.benchmark_warehouse,
                      disruption_freq=self.disruption_freq,
                      disruption_time=self.disruption_time,
                      disruption_time_std = self.disruption_time_std,
                      use_randomness=self.use_randomness,
                      use_real_pick_times=self.use_real_pick_times,
                      locations_per_category=self.locations_per_category,
                      feat_category=self.feat_category,)
            
    def step(self, action: Union[int, dict[int, int]]) -> tuple:
        # self.steps += 1
        if self.dict_type_state_reward_info:
            if not isinstance(action, dict):
                raise ValueError("Action must be a dict when dict_type_state_reward_info is True")
            return self._step_with_dict(action)
        else:
            if not isinstance(action, (int, np.integer)):
                if isinstance(action, np.ndarray):
                    if action.ndim in [0,1] and isinstance(action.item(), int):
                        action = action.item()
                    else:
                        raise ValueError("Action must be an int when dict_type_state_reward_info is False")
                else:
                    try:
                        action = action.item()
                        if not isinstance(action, int):
                            raise ValueError("Action must be an int when dict_type_state_reward_info is False")
                    except:
                        raise ValueError("Action must be an int when dict_type_state_reward_info is False")
            return self._step_no_dict(action)
        
    def _step_with_dict(self, action_dict: dict[int, int]) -> tuple:
        """Performs a step in the gym environment.
        OLD FUNCTION: NOT ACTIVELY USED ANYMORE

        Args:
            action (int): New location of the picker.

        Returns:
            tuple: state, reward, done, info for gym
        """        
        # print("test")
        for id, action in action_dict.items():
            picker = self.pickers[id]
            self._assign_location_to_picker(picker, action)
        for entity in self.entities:
            entity.reset_simpy_processes()

        start_time = self.sim.now # Used for simple penalty of runtime #TODO: Implement proper reward function
        continue_running = True
        while continue_running:
            anyof_event = simpy.events.AnyOf(
                self.sim,
                [picker.path_finished_event for picker in self.pickers]
            )
            self.sim.run(until=anyof_event)
            triggered_picker_id = int(np.where([picker.path_finished_event.triggered
                                                for picker in self.pickers])[0])
            triggered_picker = self.pickers[triggered_picker_id]
            if not (self.racks[triggered_picker.location].items != [] and
                ("AMR" in list(zip(*self.racks[triggered_picker.location].items))[0])):
                 continue_running = False
            else:
                for entity in self.entities:
                    entity.reset_simpy_processes()
        state_dict, reward_dict, done_dict, info_dict = {}, {}, {}, {}
        
        for picker in self.pickers:
            if picker.path_finished_event.triggered:
                
                state = self._create_observation(picker)
                reward = - (self.sim.now - start_time)
                if self.milp_compatible:
                    done = self.steps >= self.max_steps_end
                else:
                    done = self.steps >= self.max_steps
                mask = self.get_mask()
                location_index = 1
                max_len = max([len(amr.path) for amr in self.amrs])
                while not np.any(mask != 0) and location_index <= max_len:
                    mask = self.get_mask(location_index=location_index)
                info = {"mask": mask, "previous_mask": self.mask}
                self.mask = mask
                
                
                state_dict[picker.id] = state
                reward_dict[picker.id] = reward
                done_dict[picker.id] = done
                info_dict[picker.id] = info
        return state_dict, reward_dict, done_dict, info_dict
    
    def _assign_location_to_picker(self, picker: Picker, destination: int) -> None:
        path = self.find_shortest_path(picker.location,
                                       destination, unidirectional=False)
        picker.destination = destination
        picker.path = deque(path)
    
    def _step_no_dict(self, action: int) -> tuple:
        """Performs a step in the gym environment.

        Args:
            action (int): New location of the picker.

        Returns:
            tuple: state, reward, done, info for gym
        """        
        active_picker = self.pickers[self.active_picker_id]
        self._assign_location_to_picker(active_picker, action)
        for entity in self.entities:
            entity.reset_simpy_processes()
        start_time = self.sim.now
        new_action = False
        while not new_action:
            try:
                continue_running = True
                while continue_running:
                    anyof_event = simpy.events.AnyOf(
                        self.sim,
                        [picker.path_finished_event for picker in self.pickers]
                    )
                    self.sim.run(until=anyof_event)
                    triggered_picker_id = int(np.where([picker.path_finished_event.triggered
                                                        for picker in self.pickers])[0])
                    triggered_picker = self.pickers[triggered_picker_id]
                    if not (self.racks[triggered_picker.location].items != [] and
                        ("AMR" in list(zip(*self.racks[triggered_picker.location].items))[0])):
                        continue_running = False
                    else:
                        for entity in self.entities:
                            entity.reset_simpy_processes()
            except RuntimeError as e:
                print(f"{self.steps=} \n {self.sim.now=} \n {[picker.destination for picker in self.pickers]} \n {[picker.path for picker in self.pickers]} \n {[picker.location for picker in self.pickers]} \n {[amr.destination for amr in self.amrs]} \n {[amr.path for amr in self.amrs]} \n {[amr.location for amr in self.amrs]}")
            if self.milp_compatible:
                if self.steps >= self.max_steps_end:
                    break
            elif self.steps >= self.max_steps:
                break
            
            try:
                active_picker_id = int(
                    np.where([picker.path_finished_event.triggered
                            for picker in self.pickers])[0]
                    )
            except TypeError:
                active_picker_id = int(np.where([picker.path_finished_event.triggered
                                       for picker in self.pickers])[0][0])
            active_picker = self.pickers[active_picker_id]
            
            mask = self.get_mask(active_picker_location=active_picker.location)
            location_index = 1
            max_len = max([len(amr.pickrun) for amr in self.amrs])
            while not np.any(mask != 0) and location_index <= max_len:
                mask = self.get_mask(location_index=location_index, active_picker_location=active_picker.location)
                location_index += 1
            if np.any(mask != 0):
                new_action = True
                active_picker.idle = False
            else:
                new_location = np.argpartition(
                    self.distance_matrix_bi[active_picker.location],
                    kth=1
                )[1]
                self._assign_location_to_picker(active_picker, new_location)
                for entity in self.entities:
                    entity.reset_simpy_processes()
                active_picker.idle = True
            
        self.active_picker_id = int(
                np.where([picker.path_finished_event.triggered
                          for picker in self.pickers])[0]
        )
        state = self._create_observation(self.pickers[self.active_picker_id])
        reward = float(- (self.sim.now - start_time))
        new_fairness = - np.std(list(self.pick_counter.values()))
        reward_fairness  = new_fairness - self.current_fairness
        self.current_fairness = new_fairness
        reward = np.asarray([reward /40, reward_fairness])
        if self.milp_compatible:
            done = self.steps >= self.max_steps_end
        else:
            done = self.steps >= self.max_steps
        mask = self.get_mask(active_picker_location=active_picker.location)
        # If no "free actions" are available,
        # Create a mask with the next locations of the amrs
        location_index = 1
        max_len = max([len(amr.pickrun) for amr in self.amrs])
        while not np.any(mask != 0) and location_index <= max_len:
            mask = self.get_mask(location_index=location_index, active_picker_location=active_picker.location)
            location_index += 1
  
        info = {"mask": mask, "previous_mask": self.mask}

        if done:
            fairness_per_step = self.current_fairness / self.steps
            reward_fairness += sum([fairness_per_step * 0.995 ** i for i in range(1000)])
            reward = np.asarray([reward[0], reward_fairness])
            self.reward_history["time"].append(self.sim.now)
            self.reward_history["fairness"].append(self.current_fairness)
            self.reward_history["workload_values"].append(list(self.pick_counter.values()))
        info["obj"] = reward # For use in mopg code
        info["now"] = self.sim.now
        info["current_fairness"] = self.current_fairness
        info["workload_values"] = list(self.pick_counter.values())
        return state, reward, done, False, info
        
    def run_benchmark(self, seed=None) -> None:
        if not self.fixed_seed is None:
            seed = self.fixed_seed
        self._reset_with_seed(seed=seed)
        if self.milp_compatible:
            self.max_steps_end = self.max_steps - len([item for pickrun in self.pickruns for item in pickrun])
            self.pickruns = deque([])
        for entity in self.entities:
            entity.reset_simpy_processes()
        for picker in self.pickers:
            picker.initial_destination = picker.destination
        
        anyof_done_event = simpy.events.AnyOf(
            self.sim,
            [picker.done_event for picker in self.pickers]
        )
        self.sim.run(until=anyof_done_event)
        
    def get_mask(self, location_index: int = 0, active_picker_location: int = -1) -> np.ndarray:
        """Returns a mask of available actions.

        Returns:
            np.array: The mask, with 1 for available actions and 0
            for unavailable actions.
        """        
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if location_index == 0:

            mask.put([amr.destination for amr in self.amrs if
                    (amr.destination is not None) and (amr.destination not in
                    [picker.destination for picker in self.pickers
                     if picker.destination != int(active_picker_location)])], 1)
            if (not self.only_current_destinations_available
                and any([picker.destination in [
                            amr.destination for amr in self.amrs
                            ]
                         for picker in self.pickers
                         if picker.destination != int(active_picker_location)])):
                mask.put([amr.pickrun[0][0] for amr in self.amrs if
                          (len(amr.pickrun) >= 1) and (amr.pickrun[0][0] not in
                                                       [picker.destination for picker in self.pickers
                                                        if picker.destination != int(active_picker_location)])
                          and (amr.destination 
                               in [picker.destination for picker in self.pickers
                                   if picker.destination != int(active_picker_location)]
                                   )], 1)

        else:
            mask.put([amr.pickrun[location_index - 1][0] for amr in self.amrs if
                      (len(amr.pickrun) >= location_index) and
                      (amr.pickrun[location_index - 1][0] not in
                      [picker.destination for picker in self.pickers
                       if picker.destination != int(active_picker_location)])
                     and (not amr.pickrun[location_index - 1] is None)], 1)
        return mask
    
    def find_shortest_path(self, start: int, end: int,
                           unidirectional: bool) -> list:
        """Finds the shortest path between two nodes.

        Args:
            start (int): Start node.
            end (int): End node.

        Returns:
            list: Path between start and end. Contains indices of nodes.
        """
        if unidirectional:        
            _, predecessors = shortest_path(
                self.adjacency_matrix_uni,
                return_predecessors=True,
                indices=start
            )
        else:
            _, predecessors = shortest_path(
                self.adjacency_matrix_bi,
                return_predecessors=True,
                indices=start
            )
        return self._get_path_from_predecessors(predecessors, end)
    
    def save_trajectories(
        self,
        folder_path: Union[str, os.PathLike] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_trajectories"),
        id: Union[str, int] = 0) -> None:
        """Saves the trajectories of the pickers and AMRs to a pickle file.

        Args:
            folder_path (Union[str, os.PathLike], optional): Folder in which to
            save the files. Defaults to path_of_current_file/saved_trajectories.
            
            id (Union[str, int], optional): _description_. Defaults to 0.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(os.path.join(folder_path, f"amr_trajectories_{id}.pkl"), "wb") as f:
            pickle.dump([amr.visited_locations for amr in self.amrs], f)
        with open(os.path.join(folder_path, f"picker_trajectories_{id}.pkl"), "wb") as f:
            pickle.dump([picker.visited_locations for picker in self.pickers], f)
        
    def get_neighbors(self, node: int) -> np.array:
        """Gets the neighbors of a node.

        Args:
            node (int): The node for which to get the neighbors.

        Returns:
            np.array: The neighbors of the node.
        """
        return self.adjacency_matrix[[node],:].nonzero()[1]
    
    def find_coordinate_from_location(self, location) -> tuple[int, int]:
        """Finds the coordinates of a location in the warehouse.

        Returns:
            tuple[int, int]: x- and y-coordinate of the location.
        """
        x = location % self.y
        y = location // self.y
        if self.use_warehouse_coordinates:
            x = self.coordinates_x[location]
            y = self.coordinates_y[location]
        return (x, y) 
    
    def find_aisle_from_location(self, location) -> int:
        if self.two_sided_aisles:
            return location // self.y // 2
        else:
            return location // self.y
        
    def _get_expected_amr_wait_times(self):
        amr_wait_times = np.zeros(len(self.amrs))
        for pick_id in np.where([picker.interrupt_location=="pick" for picker in self.pickers])[0]:
            picker = self.pickers[pick_id]
            amrs_in_rack = list(filter(lambda x: x[0] == 'AMR', self.racks[picker.location].items))
            try:
                picked_amr_id = amrs_in_rack[0][1]
            except IndexError:
                try:
                    picked_amr_id = [amr.id for amr in self.amrs if amr.location == picker.location][0]
                except IndexError:
                    continue
            if not self.use_randomness or not self.use_real_pick_times:
                expected_pick_time_total = self.picking_time
            else:
                expected_pick_time_total = picker._gather_expected_real_pick_time()
            expected_pick_time_left = max(expected_pick_time_total + picker.pick_start_time - self.sim.now, 0)
            amr_wait_times[picked_amr_id] = expected_pick_time_left 
            if len(amrs_in_rack) > 1:
                current_time = expected_pick_time_left
                for i in range(1, len(amrs_in_rack)):
                    amr_id = amrs_in_rack[i][1]
                    amr = self.amrs[amr_id]
                    if not self.use_randomness or not self.use_real_pick_times:
                        expected_pick_time_this_amr = self.picking_time
                    else:    
                        expected_pick_time_this_amr = calculate_expected_pick_time(
                            amr.freq_at_destination,
                            self.volume_weight_info[amr.location][0],
                            self.volume_weight_info[amr.location][1],
                        )
                    current_time += expected_pick_time_this_amr
                    amr_wait_times[amr_id] = current_time

        for amr_id in np.where([amr.interrupt_location=="pick" for amr in self.amrs])[0]:
            amr = self.amrs[amr_id]
            if not self.use_randomness or not self.use_real_pick_times:
                expected_pick_time_total = self.picking_time
            else:
                expected_pick_time_total = calculate_expected_pick_time(
                    amr.freq_at_destination,
                    self.volume_weight_info[amr.location][0],
                    self.volume_weight_info[amr.location][1],
                )
                    
            expected_pick_time_left = max(expected_pick_time_total + amr.pick_start_time - self.sim.now, 0)
            amr_wait_times[amr_id] = expected_pick_time_left

            other_amrs_in_rack = list(filter(lambda x: x[0]== "AMR" and x[1] != amr_id, self.racks[self.amrs[amr_id].location].items))
            if len(other_amrs_in_rack) > 0:
                current_time = expected_pick_time_left
                for i in range(len(other_amrs_in_rack)):
                    amr_id  = other_amrs_in_rack[i][1]
                    amr = self.amrs[amr_id]
                    if not self.use_randomness or not self.use_real_pick_times:
                        expected_pick_time_this_amr = self.picking_time
                    else:    
                        expected_pick_time_this_amr = calculate_expected_pick_time(
                            amr.freq_at_destination,
                            self.volume_weight_info[amr.location][0],
                            self.volume_weight_info[amr.location][1],
                        )
                    current_time += expected_pick_time_this_amr
                    amr_wait_times[amr_id] = current_time

        remaining_amr_ids = np.where(amr_wait_times == 0)[0]
        for amr_id in remaining_amr_ids:
            amr = self.amrs[amr_id]
            if amr.waiting:
                location_of_picker_going_here = [picker.location for picker in self.pickers if picker.destination == amr.location]
                # IF a picker is going there, add the travel_time + pick_time
                if len(location_of_picker_going_here) > 0:
                    location = location_of_picker_going_here[0]
                    expected_travel_time = self.distance_matrix_bi[location, amr.location] / self.picker_speed
                    if not self.use_randomness or not self.use_real_pick_times:
                        exp_pick_time = self.picking_time
                    else:
                        exp_pick_time = calculate_expected_pick_time(
                            amr.freq_at_destination,
                            self.volume_weight_info[amr.location][0],
                            self.volume_weight_info[amr.location][1],
                        )
                    amr_wait_times[amr_id] = exp_pick_time + expected_travel_time
                # Otherwise, give a large value
                else:
                    if not self.use_randomness or not self.use_real_pick_times:
                        amr_wait_times[amr_id] = 50
                    else:
                        amr_wait_times[amr_id] = 500
            elif amr.interrupt_location in ["drive", "overtake"]:
                if not self.use_randomness or not self.use_real_pick_times:
                    exp_pick_time = self.picking_time
                else:
                    exp_pick_time = calculate_expected_pick_time(
                        amr.freq_at_destination,
                        self.volume_weight_info[amr.destination][0],
                        self.volume_weight_info[amr.destination][1],
                    )
                amr_wait_times[amr_id] = exp_pick_time
        return amr_wait_times
    
    def _get_expected_picker_times(self, current_picker_id):
        picker_wait_times = np.zeros(len(self.pickers))
        for pick_id in np.where([picker.interrupt_location=="pick" for picker in self.pickers])[0]:
            picker = self.pickers[pick_id]
            if not self.use_randomness or not self.use_real_pick_times:
                expected_pick_time_total = self.picking_time
            else:
                expected_pick_time_total = picker._gather_expected_real_pick_time()
            expected_pick_time_left = max(expected_pick_time_total + picker.pick_start_time - self.sim.now, 0)
            picker_wait_times[pick_id] = expected_pick_time_left
            amrs_in_rack = list(filter(lambda x: x[0] == 'AMR', self.racks[picker.location].items))
            if len(amrs_in_rack) > 1:
                for i in range(1, len(amrs_in_rack)):
                    amr_id  = amrs_in_rack[i][1]
                    amr = self.amrs[amr_id]
                    if not self.use_randomness or not self.use_real_pick_times:
                        expected_pick_time_this_amr = self.picking_time
                    else:    
                        expected_pick_time_this_amr = calculate_expected_pick_time(
                            amr.freq_at_destination,
                            self.volume_weight_info[amr.location][0],
                            self.volume_weight_info[amr.location][1],
                        )
                    picker_wait_times[pick_id] += expected_pick_time_this_amr
                    
        
        for amr_id in np.where([amr.interrupt_location=="pick" for amr in self.amrs])[0]:
            amr = self.amrs[amr_id]
            if not self.use_randomness or not self.use_real_pick_times:
                expected_pick_time_total = self.picking_time
            else:
                expected_pick_time_total = calculate_expected_pick_time(
                    amr.freq_at_destination,
                    self.volume_weight_info[amr.location][0],
                    self.volume_weight_info[amr.location][1],
                )
            expected_pick_time_left = max(expected_pick_time_total + amr.pick_start_time - self.sim.now, 0)
            picker_id = [picker.id for picker in self.pickers
                         if picker.location == amr.location 
                         and picker.destination == picker.location][0]
            picker_wait_times[picker_id] += expected_pick_time_left
            other_amrs_in_rack = list(filter(lambda x: x[0]== "AMR" and x[1] != amr_id, self.racks[self.amrs[amr_id].location].items))
            if len(other_amrs_in_rack) > 0:
                for i in range(len(other_amrs_in_rack)):
                    amr_id = other_amrs_in_rack[i][1]
                    amr = self.amrs[amr_id]
                    if not self.use_randomness or not self.use_real_pick_times:
                        expected_pick_time_this_amr = self.picking_time
                    else:    
                        expected_pick_time_this_amr = calculate_expected_pick_time(
                            amr.freq_at_destination,
                            self.volume_weight_info[amr.location][0],
                            self.volume_weight_info[amr.location][1],
                        )
                    picker_wait_times[picker_id] += expected_pick_time_this_amr
        remaining_pickers_ids = np.where(picker_wait_times == 0)[0]
        for picker_id in remaining_pickers_ids:
            if picker_id != current_picker_id:
                if not self.use_randomness or not self.use_real_pick_times:
                    picker_wait_times[picker_id] = self.picking_time
                else:
                    picker = self.pickers[picker_id]
                    picker_wait_times[picker_id] = \
                        self._get_exp_waiting_for_travel_picker_real_pick_times(picker)
        return picker_wait_times
            
    
    def _get_exp_waiting_for_travel_picker_real_pick_times(self, picker: Picker):
        pick_estimate = False
        try:
            picked_amr_id = next(
                entity for entity in self.racks[picker.location].items
                if entity[0] == "AMR"
            )[1]
            picked_amr = self.amrs[picked_amr_id]
        
        except StopIteration:
            amrs_going = [amr for amr in self.amrs if 
                          amr.destination == picker.destination]
            if len(amrs_going) > 0:
                picked_amr = amrs_going[0]
            else:
                pick_estimate = True
        if not pick_estimate:
            pick_time = calculate_expected_pick_time(
                picked_amr.freq_at_destination,
                self.volume_weight_info[picker.destination][0],
                self.volume_weight_info[picker.destination][1]
            )
        else:
            pick_time = calculate_expected_pick_time(
                1,
                self.volume_weight_info[picker.destination][0],
                self.volume_weight_info[picker.destination][1]
            )
        return pick_time
    
    def _get_summed_location_pick_times(
        self, locations_with_times: list[tuple[int, float]]
        ) -> tuple[tuple[int], tuple[float]]:
        exp_pick_times_per_location = [
            (key, sum(value for _, value in group)) for key, group in 
            itertools.groupby(
                sorted(locations_with_times,key=operator.itemgetter(0)),
                key=operator.itemgetter(0)
            )
        ]
        locations, exp_pick_times = zip(*exp_pick_times_per_location)
        return locations, exp_pick_times
    
    def _get_location_workloads_from_amr_loc_freqs(
            self, amr_loc_freqs: list[tuple[int, int]]) -> dict[int, float]:
        location_frequency_tuples = [
            (location, sum(workload for _, workload in key_workload))
            for location, key_workload in itertools.groupby(
                sorted(amr_loc_freqs,
                       key=operator.itemgetter(0)),
                operator.itemgetter(0))
        ]
        waiting_workloads = {loc: freq*self.volume_weight_info[loc, 1]
                             for loc, freq in location_frequency_tuples}
        return waiting_workloads
        
    def _create_observation(self, picker: Picker) -> GraphInstance:
        """_summary_

        Args:
            picker (Picker): The picker for which the decision is made

        Returns:
            GraphInstance: The state representation.
        """
        if not self.bipartite_state:
            # First, node feature of controlled picker location
            if self.two_sided_aisles:
                node_features = np.zeros((self.x * self.y * 2, self.num_node_features))
            else:
                node_features = np.zeros((self.x * self.y, self.num_node_features))
            if self.feat_category != "fair":
                node_features[picker.location, 0] = 1 #DISCUSSED
                # Node features of AMR locations
                node_features[[amr.location for amr in self.amrs], 1] = 1 #DISCUSSED
                # Node features of distances of possible locations to controlled picker
                node_features[:, 2] = self.distance_matrix_bi[picker.location] #DISCUSSED
                # Node feature of distance of AMRs to their respective destinations (at node of destination)
                # 5000 if no amr going there
                # print([len(amr.path) for amr in self.amrs])
                node_features[:, 3] = -10
                amr_destinations = [amr.destination for amr in self.amrs if amr.destination is not None]
                amr_distances = [self.distance_matrix_uni[amr.location, amr.destination]
                                for amr in self.amrs if amr.destination is not None]
                destinations_with_distances = zip(amr_destinations, amr_distances)
                destination_dist_dict = {}
                for destination, distance in destinations_with_distances:
                    if destination in destination_dist_dict:
                        destination_dist_dict[destination] = min(destination_dist_dict[destination],
                                                                distance)
                    else:
                        destination_dist_dict[destination] = distance      
                node_features[
                    list(destination_dist_dict.keys()), 3
                ] = list(destination_dist_dict.values()) #DISCUSSED
                
                # Node feature of distance of AMR to the respective next location in their pickrun
                # (= distance to the next destination + distance between this destination and the next one)
                node_features[:, 4] = -10 # DISCUSSED
                amr_next_destinations = [amr.pickrun[0][0] for amr in self.amrs
                                        if amr.pickrun != deque([])]
                amr_distances = [self.distance_matrix_uni[amr.location, amr.destination]
                                + self.distance_matrix_uni[amr.destination, amr.pickrun[0][0]]
                                for amr in self.amrs if
                                (amr.destination is not None and amr.pickrun != deque([]))]
                #### For expected next picking time#####
                amr_expected_waiting_times = self._get_expected_amr_wait_times()
                amr_expected_waiting_times = [time for i, time in enumerate(amr_expected_waiting_times) if
                                            self.amrs[i].pickrun != deque([])]
                amr_next_expected_picking_time = [amr_expected_waiting_times[amr_id]
                                                + amr_distances[amr_id] / self.amr_speed
                                                for amr_id in range(len(amr_distances))]
                ########### If we do not want, comment out two lines above and below change to amr_distances
                
                next_destinations_with_distances = zip(amr_next_destinations, amr_next_expected_picking_time)
                next_destination_dist_dict = {}
                for destination, distance in next_destinations_with_distances:
                    if destination in next_destination_dist_dict:
                        next_destination_dist_dict[destination] = min(next_destination_dist_dict[destination],
                                                                    distance)
                    else:
                        destination_dist_dict[destination] = distance
                node_features[
                    list(next_destination_dist_dict.keys()), 4
                ] = list(next_destination_dist_dict.values())
                
                # Similar for 2 destinations ahead
                node_features[:, 5] = -10 # DISCUSSED
                amr_next_destinations = [amr.pickrun[1][0] for amr in self.amrs
                                        if len(amr.pickrun) > 1]
                amr_distances = [self.distance_matrix_uni[amr.location, amr.destination]
                                + self.distance_matrix_uni[amr.destination, amr.pickrun[0][0]]
                                + self.distance_matrix_uni[amr.pickrun[0], amr.pickrun[1][0]]
                                for amr in self.amrs if
                                (amr.destination is not None and len(amr.pickrun) > 1)]
                #### For expected next picking time#####
                amr_expected_waiting_times = self._get_expected_amr_wait_times()
                if not self.use_randomness or not self.use_real_pick_times:
                    amr_next_expected_picking_time = [self.picking_time for amr in self.amrs
                                                if amr.destination is not None and len(amr.pickrun) > 1]
                else:
                    amr_next_expected_picking_time = [
                        calculate_expected_pick_time(amr.pickrun[0][1],
                            self.volume_weight_info[amr.pickrun[0][0]][0],
                            self.volume_weight_info[amr.pickrun[0][0]][1],
                        )
                        for amr in self.amrs if len(amr.pickrun) > 1
                    ]
                                                    
                amr_expected_waiting_times = [time for i, time in enumerate(amr_expected_waiting_times) if
                                            len(self.amrs[i].pickrun) > 1]
                amr_next_expected_picking_time = [amr_expected_waiting_times[amr_id]
                                                + amr_next_expected_picking_time[amr_id]
                                                + amr_distances[amr_id] / self.amr_speed
                                                for amr_id in range(len(amr_distances))]
                ########## If we do not want, comment out two lines above and below change to amr_distances
                next_destinations_with_distances = zip(amr_next_destinations, amr_next_expected_picking_time)
                next_destination_dist_dict = {}
                for destination, distance in next_destinations_with_distances:
                    if destination in next_destination_dist_dict:
                        next_destination_dist_dict[destination] = min(next_destination_dist_dict[destination],
                                                                    distance)
                    else:
                        destination_dist_dict[destination] = distance
                node_features[
                    list(next_destination_dist_dict.keys()), 5
                ] = list(next_destination_dist_dict.values())
                # Node feature of other picker locations
                node_features[ 
                    [picker.location for picker in self.pickers
                    if picker.id != picker.id], 6 #DISCUSSED
                ] = 1
                # Node feature of picker destinations, with distance to destination
                # (i.e. 5000 if no picker goes there, distance if it does)
                node_features[:, 7] = -10 # DISCUSSED
                other_picker_destinations = [other_picker.destination 
                                        for other_picker in self.pickers
                                        if (other_picker.destination is not None
                                            and other_picker.id != picker.id)]
                other_picker_distances = [self.distance_matrix_bi[other_picker.location,
                                                                other_picker.destination]
                                        for other_picker in self.pickers
                                        if (other_picker.destination is not None
                                            and other_picker.id != picker.id)]
                destinations_with_distances = zip(other_picker_destinations, 
                                                other_picker_distances)
                picker_destination_dist_dict = {}
                for destination, distance in destinations_with_distances:
                    if destination in picker_destination_dist_dict:
                        picker_destination_dist_dict[destination] = min(picker_destination_dist_dict[destination],
                                                                    distance)
                    else:
                        picker_destination_dist_dict[destination] = distance
                node_features[
                    list(picker_destination_dist_dict.keys()), 7
                ] = list(picker_destination_dist_dict.values())
                # node_features[
                #     [other_picker.destination for other_picker in self.pickers
                #      if (other_picker.destination is not None
                #          and other_picker.id != picker.id)], 7
                #     ] = [
                #         self.distance_matrix_bi[other_picker.location,
                #                                 other_picker.destination]
                #         for other_picker in self.pickers
                #         if (other_picker.destination is not None
                #             and other_picker.id != picker.id)
                #         ]

                # NEW FEATURES
                # current number of pickers assigned to the aisle
                picker_destination_aisles = [
                    self.find_aisle_from_location(picker.destination)
                    for picker in self.pickers if picker.destination is not None
                ]
                counts_per_aisle = np.bincount(picker_destination_aisles, minlength=self.x)
                if self.two_sided_aisles:
                    node_features[:, 8] = np.repeat(counts_per_aisle, self.y * 2) # DISCUSSED
                else:
                    node_features[:, 8] = np.repeat(counts_per_aisle, self.y)
                
                # Current number of AMRs assigned to the aisle
                amr_destination_aisles = [
                    self.find_aisle_from_location(amr.destination)
                    for amr in self.amrs if amr.destination is not None
                ]
                counts_per_aisle = np.bincount(amr_destination_aisles, minlength=self.x)
                if self.two_sided_aisles:
                    node_features[:, 9] = np.repeat(counts_per_aisle, self.y * 2) #DISCUSSED
                else:
                    node_features[:, 9] = np.repeat(counts_per_aisle, self.y)
                
                
                # Feature of minimum distance of other picker to the destination
                node_features[:, 10] = np.min( # DISCUSSED
                    np.row_stack([self.distance_matrix_bi[other_picker.location,
                                                        other_picker.destination]
                                + self.distance_matrix_bi[other_picker.destination]
                                for other_picker in self.pickers
                                if other_picker.id != picker.id]),
                    axis=0
                )
                
                # Feature of how far in aisle the location is 
                # (as percentage of nr of locations in aisle)
                if self.use_warehouse_coordinates:
                    aisle_depth = self.x * 1.4 # TODO Make parameter for this value
                else:
                    aisle_depth = self.x
                node_features[:, 11] = [
                    self.find_coordinate_from_location(loc)[0] / aisle_depth
                    if self.find_aisle_from_location(loc) % 2 == 0
                    else 1 - self.find_coordinate_from_location(loc)[0] / aisle_depth
                    for loc in range(len(node_features))
                ]
                node_features[:, 12] = [self.find_aisle_from_location(loc) / self.x
                                        for loc in range(len(node_features))]
                
                # Minimum Distance (and second minimum) of the next destination of an
                # AMR going to this location
                location_min_next_dist_dict = {}
                location_2nd_min_next_dist_dict = {}
                next_loc_min_dist_dict = {}
                next_loc_2nd_min_dist_dict = {}
                for location in [amr.destination for amr in self.amrs]:
                    distances = []
                    next_distances = []
                    for amr in self.amrs:
                        if amr.destination == location and len(amr.pickrun) > 0:
                            distances.append(self.distance_matrix_uni[location, amr.pickrun[0][0]])
                            if len(amr.pickrun) > 1:
                                next_distances.append(self.distance_matrix_uni[location, amr.pickrun[0][0]] + self.distance_matrix_uni[amr.pickrun[0][0], amr.pickrun[1][0]])
                    if len(distances) > 0:
                        min_dist = min(distances)
                        location_min_next_dist_dict[location] = min_dist
                        distances.remove(min_dist)
                    if len(distances) > 0:
                        location_2nd_min_next_dist_dict[location] = min(distances)
                    if len(next_distances) > 0:
                        min_dist = min(next_distances)
                        next_loc_min_dist_dict[location] = min_dist
                        next_distances.remove(min_dist)
                    if len(next_distances) > 1:
                        next_loc_2nd_min_dist_dict[location] = min(next_distances)
                            
                #BELOW 4 ALL DISCUSSED            
                node_features[list(location_min_next_dist_dict.keys()), 13] = list(location_min_next_dist_dict.values())
                node_features[list(location_2nd_min_next_dist_dict.keys()), 14] = list(location_2nd_min_next_dist_dict.values())
                node_features[list(next_loc_min_dist_dict.keys()), 15] = list(next_loc_min_dist_dict.values())
                node_features[list(next_loc_2nd_min_dist_dict.keys()), 16] = list(next_loc_2nd_min_dist_dict.values())
                
                # Current number of AMRs waiting in this aisle
                amr_waiting_aisles = [
                    self.find_aisle_from_location(amr.location)
                    for amr in self.amrs if amr.location is not None
                    and amr.location == amr.destination
                ]
                counts_per_aisle_wait = np.bincount(amr_waiting_aisles, minlength=self.x)
                if self.two_sided_aisles:
                    node_features[:, 17] = np.repeat(counts_per_aisle_wait, self.y * 2) #DISCUSSED
                else:
                    node_features[:, 17] = np.repeat(counts_per_aisle_wait, self.y)
                    
                # Number of AMRs assigned to location
                node_features[:, 18] = np.bincount( #DISCUSSED
                    [amr.destination for amr in self.amrs 
                    if amr.destination is not None],
                    minlength=node_features.shape[0]
                )
                
                # Distance of closest other picker destination
                node_features[:, 19] = np.min(self.distance_matrix_bi[ #DISCUSSED
                    :, [other_picker.destination for other_picker in self.pickers
                        if other_picker.id != picker.id]
                    ], axis=1)
                
                
                # Distance of this location to the closest unserved other amr destination
                try:
                    closest_two_per_node = np.partition(
                        np.where(
                                self.distance_matrix_bi == 0,
                                np.inf,
                                self.distance_matrix_bi
                            )[:, [amr.destination for amr in self.amrs
                                if not amr.destination is None]],
                            axis=1,
                            kth=1
                    )[:, :2]
                except ValueError:
                    closest_two_per_node = np.zeros((len(node_features), 2))
                try:
                    value = np.min(closest_two_per_node, axis=1)
                    node_features[:, 20] = value if value != np.inf else -10
                except ValueError:
                    pass
                # Distance of this location to the second closest unserved other amr destination
                try:
                    value = np.max(closest_two_per_node, axis=1)
                    node_features[:, 21] = value if value != np.inf else -10
                except ValueError:
                    pass
                
                # Other picker expected times to this location
                
                # Feature of minimum expected time of other picker to the destination
                picker_expected_wait_times = self._get_expected_picker_times(current_picker_id=picker.id)
                # picker_expected_wait_times = [time for i, time in enumerate(picker_expected_wait_times)
                #                               if i != picker.id]
                node_features[:, 22] = np.min( #DISCUSSED
                    np.row_stack([self.distance_matrix_bi[other_picker.location,
                                                        other_picker.destination] / self.picker_speed
                                + self.distance_matrix_bi[other_picker.destination] / self.picker_speed
                                + picker_expected_wait_times[i]
                                for i, other_picker in enumerate(self.pickers)
                                if other_picker.id != picker.id]),
                    axis=0
                )
            
                # Expected pick time of the amrs that are waiting at this location
                # exp_pick_times_waiting_amrs = [
                #     (amr.destination,
                #     calculate_expected_pick_time(
                #         amr.freq_at_destination,
                #         self.volume_weight_info[amr.destination][0],
                #         self.volume_weight_info[amr.destination][1]
                #     )
                #     if self.use_randomness and self.use_real_pick_times
                #     else self.picking_time
                #     )
                #     for amr in self.amrs 
                #     if amr.destination is not None and amr.destination == amr.location
                # ]
                # if len(exp_pick_times_waiting_amrs) > 0:
                #     locations, exp_pick_times = self._get_summed_location_pick_times(
                #         exp_pick_times_waiting_amrs
                #     )
                #     node_features[locations, 23] = exp_pick_times
                
                # # Expected pick time of the amrs that are currently assigned to this location
                # exp_pick_times_going_amrs = [
                #     (amr.destination,
                #     calculate_expected_pick_time(
                #         amr.freq_at_destination,
                #         self.volume_weight_info[amr.destination][0],
                #         self.volume_weight_info[amr.destination][1]
                #     )
                #     if self.use_randomness and self.use_real_pick_times
                #     else self.picking_time
                #     )
                #     for amr in self.amrs
                #     if amr.destination is not None and amr.destination != amr.location
                # ]
                # if len(exp_pick_times_going_amrs) > 0:
                #     locations, exp_pick_times = self._get_summed_location_pick_times(
                #         exp_pick_times_going_amrs
                #     )
                #     node_features[locations, 24] = exp_pick_times
            
            # # Expected pick time of the amrs that are going here in the next step
            # exp_pick_times_next_step_amrs = [
            #     (amr.pickrun[0][0],
            #      calculate_expected_pick_time(
            #          amr.pickrun[0][1],
            #          self.volume_weight_info[amr.pickrun[0][0]][0],
            #          self.volume_weight_info[amr.pickrun[0][0]][1]
            #      )
            #      if self.use_randomness and self.use_real_pick_times
            #      else self.picking_time
            #     )
            #     for amr in self.amrs if len(amr.pickrun) > 0
            # ]
            # if len(exp_pick_times_next_step_amrs) > 0:
            #     locations, exp_pick_times = self._get_summed_location_pick_times(
            #         exp_pick_times_next_step_amrs
            #     )
            #     node_features[locations, 25] = exp_pick_times
            
            # #TODO: ALSO nr of amrs of these different types
            # # nr of amrs that are waiting at this location
            # if len(exp_pick_times_waiting_amrs) > 0:
            #     locs_freq_counter = Counter(elem[0] for elem in exp_pick_times_waiting_amrs) 
            #     node_features[list(locs_freq_counter.keys()), 26] = list(locs_freq_counter.values())
                
            # # nr of amrs that are currently assigned to this location
            # if len(exp_pick_times_going_amrs) > 0:
            #     locs_freq_counter = Counter(elem[0] for elem in exp_pick_times_going_amrs)
            #     node_features[list(locs_freq_counter.keys()), 27] = list(locs_freq_counter.values())
            
            # # nr of amrs that are going here in the next step
            # if len(exp_pick_times_next_step_amrs) > 0:
            #     locs_freq_counter = Counter(elem[0] for elem in exp_pick_times_next_step_amrs)
            #     node_features[list(locs_freq_counter.keys()), 28] = list(locs_freq_counter.values())
            
            
            ##### FAIRNESS FEATURES #####
            # Typical workload of a single item at this location
            if self.feat_category != "effic":
                if self.feat_category == "fair":
                    add_feat = 0
                else:
                    add_feat = 23
                node_features[:, 0 + add_feat] = self.volume_weight_info[:, 1]
                avg_workload = np.mean(list(self.pick_counter.values()))
                
                # Current workload of current picker (not pure node feature)
                node_features[:, 1+ add_feat] = self.pick_counter[picker.id] - avg_workload
                # Current minimum workload of any picker
                node_features[:, 2+ add_feat] = np.min(
                    list(self.pick_counter.values())) - avg_workload
                # Current maximum workload of any picker
                node_features[:, 3+ add_feat] = np.max(
                    list(self.pick_counter.values())) - avg_workload
                # Current 25% quantile of workload
                node_features[:, 4+ add_feat] = np.quantile(
                    list(self.pick_counter.values()), 0.25) - avg_workload
                # Current 75% quantile of workload
                node_features[:, 5+ add_feat] = np.quantile(
                    list(self.pick_counter.values()), 0.75) - avg_workload
                # # Current mean workload
                # node_features[:, 35] = avg_workload
                
                # Current workload of the picker at this location
                picker_locations = [picker.location for picker in self.pickers] #DISCUSSED
                node_features[picker_locations, 6+ add_feat] = list(self.pick_counter.values())
                
                # Current workload of the picker going to this location
                picker_destinatons = [picker.destination for picker in self.pickers] #DISCUSSED
                node_features[picker_destinatons, 7+ add_feat] = list(self.pick_counter.values())
                
                # Workload of 1 item at this location
                node_features[:, 8+ add_feat] = self.volume_weight_info[:, 1] #DISCUSSED
                
                # Workload of the picks for all AMRs that are waiting at the location
                #DISCUSSED
                locations_with_freqs = [(amr.location, amr.freq_at_destination)
                                        for amr in self.amrs
                                        if amr.location == amr.destination]
                waiting_workloads = self._get_location_workloads_from_amr_loc_freqs(
                    locations_with_freqs
                )
                node_features[list(waiting_workloads.keys()), 9+ add_feat] = list(waiting_workloads.values())
                
                # Workload of the picks for all AMRs that are going to the location
                # DISCUSSED
                destinations_with_freqs = [(amr.destination, amr.freq_at_destination)
                                        for amr in self.amrs
                                        if amr.location != amr.destination
                                        and not amr.destination is None]
                destination_workloads = self._get_location_workloads_from_amr_loc_freqs(
                    destinations_with_freqs
                )
                node_features[list(destination_workloads.keys()), 10+ add_feat] = list(destination_workloads.values())
                
                all_expected_wait_times = self._get_expected_picker_times(current_picker_id=picker.id)
                # Current workload of the closest other pickers
                expected_duration_pickers = np.row_stack([self.distance_matrix_bi[other_picker.location,
                                                        other_picker.destination] / self.picker_speed
                                                        + self.distance_matrix_bi[other_picker.destination] / self.picker_speed
                                                        + all_expected_wait_times[i]
                                                        for i, other_picker in enumerate(self.pickers)
                                                        if other_picker.id != picker.id])
                closest_pick_indices = np.argmin(expected_duration_pickers, axis=0)
                closest_pick_ids = np.where(closest_pick_indices < picker.id,
                                            closest_pick_indices, closest_pick_indices + 1)
                closest_picker_workloads = np.asarray(
                    [self.pick_counter[i] for i in closest_pick_ids])
                #DISCUSSED
                node_features[:, 11+ add_feat] = closest_picker_workloads - avg_workload
                
                # Current workload of the second closest other picker
                expected_duration_pickers[closest_pick_indices,
                                        np.arange(len(closest_pick_ids))] = np.inf
                second_closest_pick_indices = np.argmin(expected_duration_pickers, axis=0)
                second_closest_pick_ids = np.where(
                    second_closest_pick_indices < picker.id,
                    second_closest_pick_indices, second_closest_pick_indices + 1)
                second_closest_picker_workloads = np.asarray(
                    [self.pick_counter[i] for i in second_closest_pick_ids])
                #DISCUSSED
                node_features[:, 12+ add_feat] = second_closest_picker_workloads - avg_workload
            
             
            mask = self.get_mask(active_picker_location=picker.location)
            # If no available location, return mask for next locations
            location_index = 1
            max_len = max([len(amr.pickrun) for amr in self.amrs])
            while not np.any(mask != 0) and location_index <= max_len:
                mask = self.get_mask(location_index=location_index,
                                     active_picker_location=picker.location)
                location_index += 1
            picks_left = self.max_steps - self.steps if not self.milp_compatible\
                else self.max_steps_end - self.steps 
            aisle_nrs = np.asarray([self.find_aisle_from_location(loc)
                                    for loc in range(len(node_features))])
            return {"graph": GraphInstance(
                nodes=node_features,
                edges=np.zeros(self.adjacency_list_bi.shape[0], dtype=np.int64),
                edge_links=self.adjacency_list_bi
            ), "mask": mask, "picks_left": picks_left, "Aisle_nrs": aisle_nrs}
        else:
            pass
      
    def _create_adjacency_matrix(self, is_unidirectional: bool) -> scipy.sparse._arrays.csr_array:
        """Creates the adjacency matrix of the underlying warehouse graph.

        Returns:
            scipy.sparse._arrays.csr_array: The adjacency matrix in sparse form.
        """
        if not is_unidirectional:
            grid_graph = create_warehouse_grid(
                self.x, self.y, is_unidirectional=False,
                two_sided_aisles=self.two_sided_aisles,
                use_warehouse_coordinates=self.use_warehouse_coordinates
            )
            adjacency_matrix = nx.to_scipy_sparse_array(grid_graph)
        else:
            unidirectional_graph = create_warehouse_grid(
                self.x, self.y, is_unidirectional=True,
                two_sided_aisles=self.two_sided_aisles,
                use_warehouse_coordinates=self.use_warehouse_coordinates
            )
            adjacency_matrix = nx.to_scipy_sparse_array(unidirectional_graph)
        return adjacency_matrix
    
    def _create_adjacency_list(self, is_unidirectional: bool) -> np.array:
        if not is_unidirectional:
            g = nx.from_scipy_sparse_array(self.adjacency_matrix_bi)
            adjacency_list = np.array(g.edges, dtype=np.int64)
        else:
            g = nx.from_scipy_sparse_array(self.adjacency_matrix_uni)
            adjacency_list = np.array(g.edges, dtype=np.int64)
        return adjacency_list
    
    def _create_distance_matrix(self, is_unidirectional: bool) -> np.array:
        """Creates the distance matrix of the underlying warehouse graph.

        Returns:
            np.array: Distance matrix.
        """
        if is_unidirectional:
            _, predecessors = shortest_path(
                self.adjacency_matrix_uni,
                return_predecessors=True,
            )
            dist_matrix = construct_dist_matrix(
                self.adjacency_matrix_uni, predecessors
            )
        else:
            _, predecessors = shortest_path(
                self.adjacency_matrix_bi,
                return_predecessors=True,
            )
            dist_matrix = construct_dist_matrix(
                self.adjacency_matrix_bi, predecessors
            )
        return dist_matrix
    
    def _get_path_from_predecessors(self, predecessors: np.array, end: int) -> list:
        """Based on the predecessors array as returned by
        scipy.sparse.csgraph.shortest_path, get path from start to end.

        Args:
            predecessors (np.array): Array of predecessors as returned by scipy.sparse.csgraph.shortest_path.
            end (int): Destination node.

        Returns:
            list: Path from start to end. Contains indices of nodes.
        """
        return numba_helpers.get_path_from_predecessors(predecessors.astype("int64"), end)
    
    def _create_racks(self) -> list[simpy.FilterStore]:
        """Creates the racks in the warehouse.

        Returns:
            list[simpy.FilterStore]: The racks.
        """        
        racking = []
        for _ in range(self.action_space.n):
            rack = simpy.FilterStore(self.sim)
            racking.append(rack)  
        return racking
    
    def _create_pickrun(self) -> deque[tuple]:
        pick_run_locations = list(self.np_random.choice(range(self.action_space.n),
                                           self.np_random.integers(
                                               self.pickrun_len_minimum,
                                               self.pickrun_len_maximum,
                                               size=1), replace=False))
        # print(pick_run_locations, random.sample(range(self.action_space.n),
        #                                    random.randrange(15, 25)))
        pick_run_with_coordinates = [
            (location, 
             self.find_coordinate_from_location(location)[0],
             self.find_aisle_from_location(location)
             ) for location in pick_run_locations
        ]
        pick_run_with_coordinates.sort(key=operator.itemgetter(2,1))
        
        # Sort row ascending or descending based on coordinate
        i = 0
        sorted_pick_run = []
        for y in sorted(list(set(x[2] for x in pick_run_with_coordinates))):
            candidates = []
            while pick_run_with_coordinates[i][2] == y:
                candidates.append(pick_run_with_coordinates[i])
                if y % 2 == 0:
                    candidates.sort(key=operator.itemgetter(1))
                else:
                    candidates.sort(key=operator.itemgetter(1), reverse=True)
                i +=1
                if i >= len(pick_run_with_coordinates):
                    break
            sorted_pick_run.extend(candidates)
        
        item_counts = sample_quantities(len(sorted_pick_run), self.np_random)
        new_pickrun = list(map(operator.itemgetter(0), sorted_pick_run))
        new_pickrun_with_counts = deque(zip(new_pickrun, item_counts))
        return new_pickrun_with_counts
    
    def pregenerate_pickruns(self, nr_amrs) -> deque: 
        pickruns = []
        total_picks = 0
        while total_picks < self.max_steps:
            new_pickrun = self._create_pickrun()
            # ENSURE That the initial pickruns are diverse
            if self.ensure_diverse_initial_pickruns:
                if len(pickruns) < nr_amrs:
                    new_pickrun = deque(
                        list(new_pickrun)[int(float(self.np_random.random())
                                        * (len(new_pickrun) -3)):
                                    ]
                    )
            pickruns.append(new_pickrun)
            total_picks += len(pickruns[-1])
        i = 1
        while total_picks > self.max_steps:
            try:
                pickruns[-i].pop()
            except IndexError:
                i = 1
                pickruns[-i].pop()
            total_picks -= 1
            i += 1
        assert total_picks == self.max_steps
        # print(pickruns, total_picks)
        return deque(pickruns)
            

class Picker():
    """Picker that performs actions in a Warehouse environment.
    """
    def __init__(self, warehouse: Warehouse, initial_location: int, id: int) -> None:
        """Initializes the picker.

        Args:
            warehouse (Warehouse): The warehouse in which the picker operates.
            initial_location (int): The initial location of the picker.
            id (int): The id of the picker.
        """
        
        self.location = initial_location
        self.warehouse = warehouse
        self.id = id
        # Store visited locations for later analysis of the paths
        self.visited_locations: list[tuple] = [(0, np.std(list(self.warehouse.pick_counter.values())), initial_location)] # (time, fairness, location)
        # Find the x- and y-coordinates of the initial location
        self.x, self.y = self._find_coordinate_from_location()
        
        # Path to the destination
        self.path = deque()
        # The current destination of the picker
        self.destination: int = None
        self.waiting = False
        # The simpy process that model the picker's behavior
        if self.warehouse.benchmark_warehouse == False:
            self.walking_process = self.warehouse.sim.process(self.walk_path())
        else:
            self.walking_process = self.warehouse.sim.process(self.benchmark_walk())
        # The simpy event that will trigger when the picker becomes available for a new pick
        self.path_finished_event = self.warehouse.sim.event()
        self.done_event = self.warehouse.sim.event()
        
        self.idle = False
        if self.warehouse.use_randomness:
            self.picker_walk_speed = max(
                self.warehouse.np_random.normal(
                    self.warehouse.picker_speed,
                    self.warehouse.picker_speed_std
                ),
                self.warehouse.picker_speed - 3 * self.warehouse.picker_speed_std
            )
        else:
            self.picker_walk_speed = self.warehouse.picker_speed
        
        self.interrupt_time_left: Optional[float]  = None
        self.interrupt_location: Optional[str]  = None
        if not self.warehouse.disruption_freq is None and self.warehouse.disruption_freq > 0:
            self.interrupt_counter = self.warehouse.np_random.poisson(
                self.warehouse.disruption_freq
            )
        else:
            self.interrupt_counter = None
        
    def walk_path(self) -> Generator:
        """The walking process of the picker.
        
        Yields:
            Generator: Simpy events.
        """
        try:
            # Walk to the next node in the path, if there is one
            while self.path:
                if self.interrupt_location is None:
                    distance = self.warehouse.distance_matrix_bi[self.location, self.path[0]]
                    travel_time = distance / self.picker_walk_speed + \
                        float(self.warehouse.np_random.random()) * 0.005
                    self.interrupt_location = "walk"
                    time_to_yield, yield_start_time = travel_time, self.warehouse.sim.now
                    yield self.warehouse.sim.timeout(travel_time)
                    self.move()
                    self.interrupt_location = None
                elif self.interrupt_location == "walk":
                    time_to_yield, yield_start_time = self.interrupt_time_left, self.warehouse.sim.now
                    yield self.warehouse.sim.timeout(self.interrupt_time_left)
                    self.move()
                    self.interrupt_location = None
                # If we allow for picking any encountered waiting AMR,
                # Check if there is an AMR at the current location
                if self.interrupt_location is None:
                    if (self.warehouse.allow_picking_encountered_vehicles
                        and self.warehouse.racks[self.location].items != []
                        and ("AMR" in list(zip(*self.warehouse.racks[self.location].items))[0])):
                        # If so, pick it
                        # NOTE: DO NOT PLACE NEXT LINES UNTIL CONTINUE_AFTER_PICK
                        # IN A FUNCTION, IT SKIPS THE FUNCTION FOR SOME REASON
                        self.interrupt_location = "encountered_pick"
                        if self.interrupt_counter is not None and self.interrupt_counter <= 0:
                            if not self.warehouse.use_randomness:
                                pick_time = self.warehouse.disruption_time + float(self.warehouse.np_random.random()) * 0.01
                            else:
                                if self.warehouse.use_real_pick_times:
                                    pick_time = self._gather_real_pick_time()
                                    pick_time += max(
                                        self.warehouse.np_random.normal(
                                            self.warehouse.disruption_time,
                                            self.warehouse.disruption_time_std
                                        ),
                                        self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                    )
                                else:
                                    pick_time = max(
                                        self.warehouse.np_random.normal(
                                            self.warehouse.disruption_time,
                                            self.warehouse.disruption_time_std
                                        ),
                                        self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                    )
                            self.interrupt_counter = self.warehouse.np_random.poisson(
                                self.warehouse.disruption_freq
                            )
                        else:
                            if not self.warehouse.use_randomness:
                                pick_time = self.warehouse.picking_time + float(self.warehouse.np_random.random()) * 0.01
                            else:
                                if self.warehouse.use_real_pick_times:
                                    pick_time = self._gather_real_pick_time()
                                else:
                                    pick_time = max(
                                        self.warehouse.np_random.normal(
                                            self.warehouse.picking_time,
                                            self.warehouse.picking_time_std
                                        ),
                                        self.warehouse.picking_time - 3 * self.warehouse.picking_time_std
                                    )
                        time_to_yield, yield_start_time = pick_time, self.warehouse.sim.now
                        yield self.warehouse.sim.timeout(pick_time)
                        self.interrupt_location = None
                        # After the pick, remove the AMR from the rack
                        amr_id = self.warehouse.racks[self.location].get(
                            lambda x: x[0] == "AMR").value[1]
                        if self.interrupt_counter is not None:
                            self.interrupt_counter -= 1
                        self.waiting=False
                        # And send the AMR to continue its journey
                        self.warehouse.amrs[amr_id].continue_after_pick()
                elif self.interrupt_location == "encountered_pick":
                    time_to_yield, yield_start_time = self.interrupt_time_left, self.warehouse.sim.now
                    yield self.warehouse.sim.timeout(self.interrupt_time_left)
                    self.interrupt_location = None
                    # After the pick, remove the AMR from the rack
                    amr_id = self.warehouse.racks[self.location].get(
                        lambda x: x[0] == "AMR").value[1]
                    if self.interrupt_counter is not None:
                        self.interrupt_counter -= 1
                    self.waiting=False
                    # And send the AMR to continue its journey
                    self.warehouse.amrs[amr_id].continue_after_pick()
                else: # In very rare cases, the interrupt location is "pick" it has a path.
                    self.interrupt_location = None
            # If there is no path, the destination is reached, so:
            else:
                # Once the path is done, initialize an new picker speed for the next path,
                # if we use randomness
                if self.warehouse.use_randomness:
                    self.picker_walk_speed = max(
                        self.warehouse.np_random.normal(
                            self.warehouse.picker_speed,
                            self.warehouse.picker_speed_std
                        ),
                        self.warehouse.picker_speed - 3 * self.warehouse.picker_speed_std
                    )
                else:
                    self.picker_walk_speed = self.warehouse.picker_speed
                # Check if an AMR is already at the destination
                if (self.warehouse.racks[self.location].items != [] and
                    ("AMR" in 
                     list(zip(*self.warehouse.racks[self.location].items))[0])
                    and self.interrupt_location in [None, "pick"]):
                    while (self.warehouse.racks[self.location].items != [] and
                           ("AMR" in 
                            list(zip(*self.warehouse.racks[self.location].items))[0])):
                            
                        # If so, picking 
                        if self.interrupt_location in [None, "pick"]:
                            if self.interrupt_location is None:
                        # NOTE: DO NOT PLACE NEXT LINES UNTIL CONTINUE_AFTER_PICK
                        # IN A FUNCTION, IT SKIPS THE FUNCTION FOR SOME REASON
                                self.interrupt_location = "pick"
                                if self.interrupt_counter is not None and self.interrupt_counter <= 0:
                                    if not self.warehouse.use_randomness:
                                        pick_time = self.warehouse.disruption_time + float(self.warehouse.np_random.random()) * 0.01
                                    else:
                                        if self.warehouse.use_real_pick_times:
                                            pick_time = self._gather_real_pick_time()
                                            pick_time += max(
                                                self.warehouse.np_random.normal(
                                                    self.warehouse.disruption_time,
                                                    self.warehouse.disruption_time_std
                                                ),
                                                self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                            )
                                        else:
                                            pick_time = max(
                                                self.warehouse.np_random.normal(
                                                    self.warehouse.disruption_time,
                                                    self.warehouse.disruption_time_std
                                                ),
                                                self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                            )
                                    self.interrupt_counter = self.warehouse.np_random.poisson(
                                        self.warehouse.disruption_freq
                                    )
                                else:
                                    if not self.warehouse.use_randomness:
                                        pick_time = self.warehouse.picking_time + float(self.warehouse.np_random.random()) * 0.01
                                    else:
                                        if self.warehouse.use_real_pick_times:
                                            pick_time = self._gather_real_pick_time()
                                        else:
                                            pick_time = max(
                                                self.warehouse.np_random.normal(
                                                    self.warehouse.picking_time,
                                                    self.warehouse.picking_time_std
                                                ),
                                                self.warehouse.picking_time - 3 * self.warehouse.picking_time_std
                                            )

                                time_to_yield, yield_start_time = pick_time, self.warehouse.sim.now
                                self.pick_start_time = yield_start_time
                                yield self.warehouse.sim.timeout(pick_time)
                                self.interrupt_location = None
                                if self.interrupt_counter is not None:
                                    self.interrupt_counter -= 1
                            else:
                                time_to_yield, yield_start_time = self.interrupt_time_left, self.warehouse.sim.now
                                yield self.warehouse.sim.timeout(self.interrupt_time_left)
                                self.interrupt_location = None
                                if self.interrupt_counter is not None:
                                    self.interrupt_counter -= 1
                            # After the pick, remove the AMR from the rack
                            get_return = self.warehouse.racks[self.location].get(lambda x: x[0] == "AMR")
                            amr_id = get_return.value[1]
                            self.waiting=False
                            # And send the AMR to continue its journey
                            self.warehouse.amrs[amr_id].continue_after_pick()
                            # Picker is now available for a new pick 
                    yield self.path_finished_event.succeed()
                # If the AMR is not at the destination, the picker waits for the AMR
                else:
                    if not self.idle:
                        if self.warehouse.allow_picker_waiting:
                            if not self.waiting: 
                                yield self.warehouse.racks[self.location].put(("PICKER",self.id))
                                self.waiting = True
                        else:
                            yield self.path_finished_event.succeed()
                    else:
                        yield self.path_finished_event.succeed()
        # Process is interrupted when any picker becomes available for a new pick
        # And all processes are reset 
        # (This is needed to prevent "double" processes in the simpy environment)) 
        except simpy.Interrupt:
            try:
                self.yield_start_time = yield_start_time
                self.interrupt_time_left = time_to_yield + yield_start_time - self.warehouse.sim.now
            except UnboundLocalError:
                self.interrupt_time_left = None
                self.yield_start_time = None
    
    def _gather_expected_real_pick_time(self) -> float:
        try:
            picked_amr_id = next(
                entity for entity in self.warehouse.racks[self.location].items
                if entity[0] == "AMR"
            )[1]
        # Exception only occur when all picks are done and this is called in create observation
        except StopIteration:
            return self.warehouse.picking_time
        picked_amr = self.warehouse.amrs[picked_amr_id]
        pick_time = calculate_expected_pick_time(
            picked_amr.freq_at_destination,
            self.warehouse.volume_weight_info[self.location][0],
            self.warehouse.volume_weight_info[self.location][1]
        )
        return pick_time
    
    def _gather_real_pick_time(self) -> float:
        pick_time = self._gather_expected_real_pick_time()
        pick_time = self.warehouse.np_random.normal(pick_time, pick_time * 0.1)
        return pick_time
    
    def benchmark_walk(self) -> Generator:
        try:
            if not hasattr(self.warehouse, "aisle_to_loc_dict"):
                self.warehouse.aisle_to_loc_dict = {aisle: [] for aisle in range(self.warehouse.x)}
                for location in range(self.warehouse.distance_matrix_bi.shape[0]):
                    aisle = self.warehouse.find_aisle_from_location(location)
                    self.warehouse.aisle_to_loc_dict[aisle].append(location) 
            self.warehouse._assign_location_to_picker(self, self.destination)
            while self.path:
                distance = self.warehouse.distance_matrix_bi[self.location, self.path[0]]
                travel_time = distance / self.picker_walk_speed + \
                        float(self.warehouse.np_random.random()) * 0.005
                yield self.warehouse.sim.timeout(travel_time)
                self.move()
            
            if (self.warehouse.racks[self.location].items != [] and
                    ("AMR" in 
                     list(zip(*self.warehouse.racks[self.location].items))[0])
                    ):
                while (self.warehouse.racks[self.location].items != [] and
                        ("AMR" in 
                        list(zip(*self.warehouse.racks[self.location].items))[0])):
                    if self.interrupt_counter is not None and self.interrupt_counter <= 0:
                        if not self.warehouse.use_randomness:
                            pick_time = self.warehouse.disruption_time + float(self.warehouse.np_random.random()) * 0.01
                        else:
                            if self.warehouse.use_real_pick_times:
                                pick_time = self._gather_real_pick_time()
                                pick_time += max(
                                    self.warehouse.np_random.normal(
                                        self.warehouse.disruption_time,
                                        self.warehouse.disruption_time_std
                                    ),
                                    self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                )
                            else:
                                pick_time = max(
                                    self.warehouse.np_random.normal(
                                        self.warehouse.disruption_time,
                                        self.warehouse.disruption_time_std
                                    ),
                                    self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                )
                        self.interrupt_counter = self.warehouse.np_random.poisson(
                            self.warehouse.disruption_freq
                        )
                    else:
                        if not self.warehouse.use_randomness:
                            pick_time = self.warehouse.picking_time + float(self.warehouse.np_random.random()) * 0.01
                        else:
                            if self.warehouse.use_real_pick_times:
                                    pick_time = self._gather_real_pick_time()
                            else:
                                pick_time = max(
                                    self.warehouse.np_random.normal(
                                        self.warehouse.picking_time,
                                        self.warehouse.picking_time_std
                                    ),
                                    self.warehouse.picking_time - 3 * self.warehouse.picking_time_std
                                )
                    yield self.warehouse.sim.timeout(pick_time)
                    if self.interrupt_counter is not None:
                        self.interrupt_counter -= 1
                    print(f"{self.warehouse.racks[self.location].items=}")
                    get_return = self.warehouse.racks[self.location].get(lambda x: x[0] == "AMR")
                    try:
                        amr_id = get_return.value[1]
                        # And send the AMR to continue its journey
                        self.warehouse.amrs[amr_id].continue_after_pick()
                    except AttributeError:
                        pass
                    self.waiting=False

            else:
                if not self.idle:
                    if self.warehouse.allow_picker_waiting:
                        if not self.waiting: 
                            yield self.warehouse.racks[self.location].put(("PICKER",self.id))
                            self.waiting = True
                    else:
                        pass
                else:
                    pass
            while self.waiting:
                yield self.warehouse.sim.timeout(0.01)
            
            while True:
                # Check if an AMR is already at the location
                if (((self.warehouse.racks[self.location].items != [] and
                    ("AMR" in 
                        list(zip(*self.warehouse.racks[self.location].items))[0]))
                    or self.location in [amr.location for amr in self.warehouse.amrs if amr.location == amr.destination and amr.waiting])
                    and self.location not in [picker.location for picker in self.warehouse.pickers
                                              if picker.id != self.id]):
                    while ((self.warehouse.racks[self.location].items != [] and
                            ("AMR" in 
                            list(zip(*self.warehouse.racks[self.location].items))[0]))
                           or self.location in [amr.location for amr in self.warehouse.amrs if amr.location == amr.destination and amr.waiting]): 
                        if self.interrupt_counter is not None and self.interrupt_counter <= 0:
                            if not self.warehouse.use_randomness:
                                pick_time = self.warehouse.disruption_time + float(self.warehouse.np_random.random()) * 0.01
                            else:
                                if self.warehouse.use_real_pick_times:
                                    pick_time = self._gather_real_pick_time()
                                    pick_time += max(
                                        self.warehouse.np_random.normal(
                                            self.warehouse.disruption_time,
                                            self.warehouse.disruption_time_std
                                        ),
                                        self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                    )
                                else:
                                    pick_time = max(
                                        self.warehouse.np_random.normal(
                                            self.warehouse.disruption_time,
                                            self.warehouse.disruption_time_std
                                        ),
                                        self.warehouse.disruption_time - 3 * self.warehouse.disruption_time_std
                                    )
                            self.interrupt_counter = self.warehouse.np_random.poisson(
                                self.warehouse.disruption_freq
                            )
                        else:
                            if not self.warehouse.use_randomness:
                                pick_time = self.warehouse.picking_time + float(self.warehouse.np_random.random()) * 0.01
                            else:
                                if self.warehouse.use_real_pick_times:
                                    pick_time = self._gather_real_pick_time()
                                else:
                                    pick_time = max(
                                        self.warehouse.np_random.normal(
                                            self.warehouse.picking_time,
                                            self.warehouse.picking_time_std
                                        ),
                                        self.warehouse.picking_time - 3 * self.warehouse.picking_time_std
                                    )
                        yield self.warehouse.sim.timeout(pick_time)
                        if self.interrupt_counter is not None:
                            self.interrupt_counter -= 1
                        # After the pick, remove the AMR from the rack
                        get_return = self.warehouse.racks[self.location].get(lambda x: x[0] == "AMR")
                        try:
                            amr_id = get_return.value[1]
                        except AttributeError:
                            try:
                                amr_id = [amr.id for amr in self.warehouse.amrs 
                                        if amr.location == amr.destination
                                        and amr.waiting and amr.location == self.location][0]
                            except IndexError:
                                amr_id = None

                        # And send the AMR to continue its journey
                        if amr_id is not None:
                            self.warehouse.amrs[amr_id].continue_after_pick()
                        if self.warehouse.milp_compatible:
                            done = self.warehouse.steps >= self.warehouse.max_steps_end
                        else:
                            done = self.warehouse.steps >= self.warehouse.max_steps
                        if done:
                            yield self.done_event.succeed()
                             
                if self.location != self.destination and len(self.path) > 0:
                    connected_locations = self.warehouse.adjacency_matrix_bi[[self.location], :].nonzero()[1]
                    current_aisle = self.warehouse.find_aisle_from_location(self.location)
                    current_depth = self.warehouse.find_coordinate_from_location(self.location)[0]
                    for location in connected_locations:
                        location_aisle = self.warehouse.find_aisle_from_location(location)
                        location_depth = self.warehouse.find_coordinate_from_location(location)[0]
                        if location_aisle == current_aisle and location_depth == current_depth:
                            if (self.warehouse.racks[location].items != [] and
                                ("AMR" in 
                                    list(zip(*self.warehouse.racks[location].items))[0])
                                and location not in [picker.location for picker in self.warehouse.pickers
                                                            if picker.id != self.id]):
                                temp_loc = self.location
                                self.warehouse._assign_location_to_picker(self, location)
                                break

                if len(self.path) == 0:
                    current_aisle = self.warehouse.find_aisle_from_location(self.location)
                    current_depth = self.warehouse.find_coordinate_from_location(self.location)[0]
                    same_aisle_locations = self.warehouse.aisle_to_loc_dict[current_aisle]
                    candidate_locations = []
                    for location in same_aisle_locations:
                        if location != self.location:
                            location_depth = self.warehouse.find_coordinate_from_location(location)[0]
                            if self.warehouse.use_warehouse_coordinates:
                                allowed_distance = 10 * 1.4 
                            else:
                                allowed_distance = 10
                            if (location_depth <= current_depth + allowed_distance and
                                location_depth >= current_depth - allowed_distance):
                                if (self.warehouse.racks[location].items != [] and
                                    ("AMR" in 
                                        list(zip(*self.warehouse.racks[location].items))[0])
                                    and location not in [picker.location for picker in self.warehouse.pickers
                                              if picker.id != self.id]):
                                    candidate_locations.append(
                                        (location,
                                        self.warehouse.distance_matrix_bi[self.location,
                                                                        location])
                                    )
                    if len(candidate_locations) > 0:
                        sorted_candidate_locations = sorted(candidate_locations,
                                                            key=operator.itemgetter(1))
                        destination = sorted_candidate_locations[0][0]
                        self.warehouse._assign_location_to_picker(self, destination)
                    else:
                        current_aisle = self.warehouse.find_aisle_from_location(self.location)
                        current_depth = self.warehouse.find_coordinate_from_location(self.location)[0]
                        if current_aisle % 2 == 0:
                            if self.warehouse.use_warehouse_coordinates:
                                max_depth = 1.4 * (self.warehouse.y - 1)
                            else:
                                max_depth = (self.warehouse.y - 1)
                            if current_depth < max_depth + 0.0001 and current_depth > max_depth - 0.0001:
                                # find new aisle
                                self.assign_picker_to_new_aisle_benchmark(current_aisle)
                            else:
                                # Take a step in current aisle.
                                self.find_next_location_benchmark(current_depth)
                        else:
                            if current_depth < 0.0001 and current_depth > -0.0001:
                                # find new aisle
                                self.assign_picker_to_new_aisle_benchmark(current_aisle)
                            else:
                                # Take a step in current aisle.
                                self.find_next_location_benchmark(current_depth)
                        
                if len(self.path) > 0:
                    distance = self.warehouse.distance_matrix_bi[self.location, self.path[0]]
                    travel_time = distance / self.picker_walk_speed + \
                        float(self.warehouse.np_random.random()) * 0.005
                    yield self.warehouse.sim.timeout(travel_time)
                    self.move()
        except simpy.Interrupt:
            pass
                
    def assign_picker_to_new_aisle_benchmark(self, current_aisle) -> None:
        nr_aisles_diff_dict = {aisle: abs(aisle - current_aisle) for aisle in range(self.warehouse.x)}
        picker_locations = [picker.location for picker in self.warehouse.pickers]
        stranded_vehicle_count_dict = {aisle: 0 for aisle in range(self.warehouse.x)}
        stranded_vehicle_dict = {aisle: [] for aisle in range(self.warehouse.x)}
        for amr in self.warehouse.amrs:
            if amr.waiting and amr.location not in picker_locations:
                aisle = self.warehouse.find_aisle_from_location(amr.location)
                stranded_vehicle_count_dict[aisle] += 1
                stranded_vehicle_dict[aisle].append(amr)
                
        new_aisle_costs = {aisle: 1* nr_aisles_diff_dict[aisle]
                           - 1 * stranded_vehicle_count_dict[aisle]
                           for aisle in range(self.warehouse.x) if stranded_vehicle_count_dict[aisle] > 0}
        if len(new_aisle_costs) == 0:
            cheapest_aisle = max([aisle for aisle in nr_aisles_diff_dict 
                                  if nr_aisles_diff_dict[aisle] == 1])
            locations_in_aisle = self.warehouse.aisle_to_loc_dict[cheapest_aisle]
            closest_location_in_aisle = locations_in_aisle[int(
                np.argmin(
                    self.warehouse.distance_matrix_bi[
                        self.location,
                        locations_in_aisle
                    ]
                )
            )]
        else:
            cheapest_aisle = min(new_aisle_costs, key=new_aisle_costs.get) # type: ignore
            stranded_vehicle_locations =[amr.location for amr in 
                                         stranded_vehicle_dict[cheapest_aisle]]
            closest_location_in_aisle = stranded_vehicle_locations[int(
                np.argmin(
                    self.warehouse.distance_matrix_bi[
                        self.location,
                        stranded_vehicle_locations
                    ]
                )
            )]
            
        self.warehouse._assign_location_to_picker(
            self, closest_location_in_aisle
        )
                
    def find_next_location_benchmark(self, current_depth) -> None:
        neighbor_locations = self.warehouse.adjacency_matrix_uni[
            [self.location], :].nonzero()[1]

        for neighbor in neighbor_locations:
            neighbor_depth = self.warehouse.find_coordinate_from_location(neighbor)[0]
            if neighbor_depth > current_depth + 0.0001 or neighbor_depth < current_depth - 0.0001:
                self.warehouse._assign_location_to_picker(self, neighbor)
                break
                
    def pick_amr(self) -> Generator:
        #NOTE: DO NOT USE THIS FUNCTION AS IT IS SKIPED FOR SOME REASON
        yield self.warehouse.sim.timeout(self.warehouse.picking_time + float(self.warehouse.np_random.random()) * 0.01)
        # After the pick, remove the AMR from the rack
        amr_id = self.warehouse.racks[self.location].get(lambda x: x[0] == "AMR").value[1]
        self.waiting=False
        # And send the AMR to continue its journey
        self.warehouse.amrs[amr_id].continue_after_pick()
    
    def move(self) -> None:
        """Performs a single step in the path.
        """
        new_location = self.path.popleft()
        self.location = new_location
        self.x, self.y = self._find_coordinate_from_location()
        self.visited_locations.append((self.warehouse.sim.now, np.std(list(self.warehouse.pick_counter.values())), new_location))
    
    def _find_coordinate_from_location(self) -> tuple[int, int]:
        """Get the x- and y-coordinates of a location.

        Returns:
            tuple[int, int]: x- and y-coordinates.
        """
        return self.warehouse.find_coordinate_from_location(self.location)

    def reset_simpy_processes(self) -> None:
        """Resets the simpy processes of the picker.
        """
        if self.walking_process.target:
            if not self.walking_process.target.processed:
                self.walking_process.interrupt()
        self.path_finished_event =  self.warehouse.sim.event()
        if self.warehouse.benchmark_warehouse:
            self.walking_process = self.warehouse.sim.process(self.benchmark_walk())
        else:
            self.walking_process = self.warehouse.sim.process(self.walk_path())

         
class AMR():
    """AMR that performs actions in a Warehouse environment.
    """
    def __init__(self, warehouse: Warehouse, initial_location: int, id: int) -> None:
        """Initializes the AMR.

        Args:
            warehouse (Warehouse): Warehouse in which the AMR operates.
            initial_location (int): Location at which the AMR is initialized.
            id (int): ID of the AMR.
        """
        self.location = initial_location
        self.warehouse = warehouse
        self.id = id
        self.finished = False
        self.visited_locations: list[tuple] = [(0, np.std(list(self.warehouse.pick_counter.values())), initial_location)] # (time, fairness, location)
        
        self.x, self.y = self._find_coordinate_from_location()
        
        self.pickrun = deque()
        self.get_new_pickrun()
        # Ensure diversity in initial pickruns by randomly selecting a fraction of the pickrun
        if not self.warehouse.generate_pickruns_in_advance:
            # This is already done when pickruns are pregenerated
            if self.ensure_diverse_initial_pickruns:
                self.pickrun = deque(
                    list(self.pickrun)[int(float(self.warehouse.np_random.random()) * (len(self.pickrun) -3)):]
                )
                
        if self.pickrun[0][0] == self.location:
            self.pickrun.popleft()
        
        self.freq_at_destination: int = 1
        self.path = deque()
        self.get_new_path() # Initialize an AMR path
        
        self.destination: int = self.path[-1]
        # The simpy process that model the AMR's behavior
        self.driving_process = self.warehouse.sim.process(self.drive_path())
        
        # Indicators for whether an AMR is currently waiting or has finished a pick
        self.item_picked = False
        self.waiting = False
        if self.warehouse.use_randomness:
            self.amr_drive_speed = max(
                self.warehouse.np_random.normal(
                    self.warehouse.amr_speed,
                    self.warehouse.amr_speed_std
                ),
                self.warehouse.amr_speed - 3 * self.warehouse.amr_speed_std
            )
        else:
            self.amr_drive_speed = self.warehouse.amr_speed
        
        self.interrupt_time_left: Optional[float]  = None
        self.interrupt_location: Optional[str]  = None
        
    def drive_path(self) -> Generator:
        """Model the driving process of the AMR.

        Yields:
            Generator: Simpy events.
        """
        try:
            # Drive to the next node in the path, if there is one
            while self.path:
                # travel_time = random.random() * 0.2 + 0.6 # avg 0.7
                if self.interrupt_location is None:
                    distance = self.warehouse.distance_matrix_bi[self.location, self.path[0]] 
                    travel_time = distance / self.amr_drive_speed + float(self.warehouse.np_random.random()) * 0.01
                    self.interrupt_location = "drive"
                    time_to_yield, yield_start_time = travel_time, self.warehouse.sim.now
                    yield self.warehouse.sim.timeout(travel_time)
                    self.move()
                    self.interrupt_location = None
                elif self.interrupt_location == "drive":
                    time_to_yield, yield_start_time = self.interrupt_time_left, self.warehouse.sim.now
                    yield self.warehouse.sim.timeout(self.interrupt_time_left)
                    self.move()
                    self.interrupt_location = None
                # If there are AMRs waiting at the new location
                # and this location is not the destination of this AMR,
                # the AMR must overtake them
                if self.interrupt_location is None:
                    if (self.warehouse.has_congestion
                        and self.destination != self.location
                        and self.warehouse.racks[self.location].items != []
                        and "AMR" in list(
                            zip(*self.warehouse.racks[self.location].items))[0]):
                            self.interrupt_location = "overtake"
                            if not self.warehouse.use_randomness:
                                overtake_time = self.warehouse.overtake_penalty + \
                                    float(self.warehouse.np_random.random()) * 0.01
                            else:
                                overtake_time = max(
                                    self.warehouse.np_random.normal(
                                        self.warehouse.overtake_penalty,
                                        self.warehouse.overtake_penalty_std
                                    ),
                                    self.warehouse.overtake_penalty - 3 * self.warehouse.overtake_penalty_std
                                )
                            time_to_yield, yield_start_time = overtake_time, self.warehouse.sim.now
                            yield self.warehouse.sim.timeout(overtake_time)
                            self.interrupt_location = None
                elif self.interrupt_location == "overtake":
                    time_to_yield, yield_start_time = self.interrupt_time_left, self.warehouse.sim.now
                    yield self.warehouse.sim.timeout(self.interrupt_time_left)
                    self.interrupt_location = None
                    
            # If there is no path, the destination is reached, so:
            else:
                if not (self.waiting or self.finished):
                    # Check if a picker is already at the destination
                    if (self.warehouse.racks[self.location].items != [] and 
                        "PICKER" in list(zip(*self.warehouse.racks[self.location].items))[0]
                        and self.location in [picker.location for picker in self.warehouse.pickers 
                                              if picker.location == picker.destination]
                        and not self.location in [amr.location for amr in self.warehouse.amrs if amr.id != self.id and amr.location == amr.destination]):
                        # If so, picking timeout
                        if self.interrupt_location in [None, "pick"]:
                            if self.interrupt_location is None:
                                self.interrupt_location = "pick"
                                if not self.warehouse.use_randomness:
                                    picking_time = self.warehouse.picking_time + float(self.warehouse.np_random.random()) * 0.01
                                else:
                                    if self.warehouse.use_real_pick_times:
                                        picking_time = calculate_expected_pick_time(
                                            self.freq_at_destination,
                                            self.warehouse.volume_weight_info[self.location][0],
                                            self.warehouse.volume_weight_info[self.location][1],
                                        )
                                        picking_time = self.warehouse.np_random.normal(
                                            picking_time,
                                            picking_time * 0.1)
                                    else:
                                        picking_time = max(
                                            self.warehouse.np_random.normal(
                                                self.warehouse.picking_time,
                                                self.warehouse.picking_time_std
                                            ),
                                            self.warehouse.picking_time - 3 * self.warehouse.picking_time_std
                                        )

                                time_to_yield = picking_time
                                yield_start_time = self.warehouse.sim.now
                                self.pick_start_time = self.warehouse.sim.now
                                yield self.warehouse.sim.timeout(picking_time)
                                self.interrupt_location = None
                            elif self.interrupt_location == "pick":
                                time_to_yield, yield_start_time = self.interrupt_time_left, self.warehouse.sim.now
                                yield self.warehouse.sim.timeout(self.interrupt_time_left)
                                self.interrupt_location = None
                            # After the pick, continue the AMR's journey
                            self.continue_after_pick()
                            # Indicate that the AMR has finished a pick and is now available for a new pick
                            try:
                                picker_id = [picker.id for picker in self.warehouse.pickers
                                             if picker.location == self.location
                                             and picker.destination == self.location][0]
                                self.warehouse.pickers[picker_id].waiting = False
                                if self.warehouse.pickers[picker_id].interrupt_counter is not None:
                                    self.warehouse.pickers[picker_id].interrupt_counter -= 1
                            except AttributeError as e:
                                if "is not yet available" in str(e):
                                    pass
                                else: 
                                    raise e
                            
                            try:
                                yield self.warehouse.racks[self.location].get(lambda x: x[0] == "PICKER")
                                yield self.warehouse.pickers[picker_id].path_finished_event.succeed()
                            # When two AMRs arrive at the exact same time,
                            # the event is already triggered after the first picking action
                            # With random times, this won't happen, but leave it in for extra safety
                            except (RuntimeError, UnboundLocalError) as e:
                                if "has already been triggered" in str(e):
                                    pass
                                else:
                                    # Remove this if condition when the above TODO is done
                                    if "local variable 'picker_id' referenced before assignment" in str(e):
                                        pass
                                    else:
                                        raise e
                        else:
                            pass
                    # If the picker is not at the destination, the AMR waits for the picker
                    else:
                        yield self.warehouse.racks[self.location].put(("AMR",self.id))
                        self.waiting = True
                        self.interrupt_location = None
                    
                else: pass # Don't know why, but removing this gives an error
        # Process is interrupted when any picker becomes available for a new pick
        # And all processes are reset 
        # (This is needed to prevent "double" processes in the simpy environment))             
        except simpy.Interrupt:
            self.interrupt_time_left = time_to_yield + yield_start_time - self.warehouse.sim.now
    
    def continue_after_pick(self) -> None:
        """Continues the AMR's journey after a pick, by setting a new destination and path.
        """
        self.warehouse.steps += 1
        try:
            picker_at_location = [picker.id for picker in self.warehouse.pickers
                                  if picker.location == self.location
                                  and picker.destination == self.location][0]
            self.warehouse.pick_counter[picker_at_location] += \
                self.freq_at_destination * \
                self.warehouse.volume_weight_info[self.location][1]
        except IndexError:
            pass
        self.item_picked = True
        self.waiting = False
        if self.warehouse.use_randomness:
            self.amr_drive_speed = max(
                self.warehouse.np_random.normal(
                    self.warehouse.amr_speed,
                    self.warehouse.amr_speed_std
                ),
                self.warehouse.amr_speed - 3 * self.warehouse.amr_speed
            )
        else:
            self.amr_drive_speed = self.warehouse.amr_speed
        self.get_new_path()
        # Don't know anymore why the exception is needed, but it is
        try:
            self.destination = self.path[-1]
        except IndexError:
            self.destination = None
        self.driving_process = self.warehouse.sim.process(self.drive_path())
            
    def move(self) -> None:
        """Steps the AMR to the next node in the path.
        """
        new_location = self.path.popleft()
        self.location = new_location
        self.x, self.y = self._find_coordinate_from_location()
        self.visited_locations.append((self.warehouse.sim.now, np.std(list(self.warehouse.pick_counter.values())), new_location))
    
    
    def get_new_pickrun(self) -> None:
        if self.warehouse.generate_pickruns_in_advance:
            if self.warehouse.pickruns:
                new_pickrun = self.warehouse.pickruns.popleft()
            else:
                self.finished = True
                return 
        else:
            new_pickrun = self.warehouse._create_pickrun()
            if new_pickrun[0][0] == self.location:
                new_pickrun.popleft()
        self.pickrun = new_pickrun
        
    
    def get_new_path(self) -> None:
        """Gets a new path for the AMR."""
        if self.pickrun:
            new_location_freq_tuple = self.pickrun.popleft()
        else:
            self.get_new_pickrun()
            if self.finished:
                return
            new_location_freq_tuple = self.pickrun.popleft()
        self.path = deque(
            self.warehouse.find_shortest_path(self.location, new_location_freq_tuple[0],
                                              unidirectional=True)
        )
        self.freq_at_destination = new_location_freq_tuple[1]
        
    def reset_simpy_processes(self) -> None:
        """Resets the simpy processes of the AMR.
        """
        # Interrupt the driving process if it is still active
        try:
            self.driving_process.interrupt()
        except RuntimeError as e:
            pass
        self.driving_process = self.warehouse.sim.process(self.drive_path())
    
    def _find_coordinate_from_location(self) -> tuple[int, int]:
        """Finds the coordinates of a location in the warehouse.

        Returns:
            tuple[int, int]: x- and y-coordinate of the location.
        """
        return self.warehouse.find_coordinate_from_location(self.location)
           
# %%
# Testing the environment using random actions or GREEDY (=NEAREST_NEIGHBOR)
# if __name__ == "__main__":
#     NEAREST_NEIGHBOR = True
    
#     MAX_STEPS = 1000
#     NR_AISLES = 14
#     DEPTH_OF_AISLES = 14
#     warehouse = Warehouse(NR_AISLES, DEPTH_OF_AISLES, 7, 15, max_steps=MAX_STEPS,
#                           generate_pickruns_in_advance=True
#                         #   two_sided_aisles=True,
#                         #   has_congestion=True,
#                         #   use_warehouse_coordinates=True)
#     )
#     scores = []
#     for episode in range(1):
#         print(f"Episode {episode} \n")
#         state_dict, info = warehouse.reset() #Initialize state
#         done = False
#         total_reward = 0
#         i=0
#         while not done:
#             #Randomly sample an action
#             # print(f"mask: {warehouse.get_mask()}")
            
#             mask = warehouse.get_mask()
#             # print(f"{warehouse.steps=},   {mask=}\n {warehouse.pickruns=}")
#             action_dict = {}
#             for id in state_dict:
#                 if NEAREST_NEIGHBOR:
#                     sampled_action = np.argmin(np.where(mask, state_dict[id]["graph"].nodes[:, 2], np.inf))
#                 else:
#                     sampled_action = warehouse.action_space.sample(mask=mask)
#                 action_dict[id] = sampled_action
#                 mask[sampled_action] = 0
#             # print(f"Walking to location at step {i} at time {warehouse.sim.now}: ", action_dict)
#             state_dict, reward_dict, done_dict, info_dict = warehouse.step(action_dict) #Perform the action
#             # print(f"{done_dict=}")
#             # print(info_dict)
#             total_reward += list(reward_dict.values())[0] # TODO check if we keep separate reward dict or not
#             if i == MAX_STEPS -1:
#                 done = True
#             if i % 100 == 0:
#                 print(i)
#             i+=1
#             # print(f"We tried adding item {action}, this has a weight of {state[0, action]} and added {reward}, leading to total reward so far of {total_reward} and total weight of {state[1, -1]}")
#         print(f"Final reward = {total_reward}")
#         scores.append(total_reward)
#     print(f"Average score: {np.mean(scores)}, Std: {np.std(scores)}, CI: {DescrStatsW(scores).tconfint_mean()}")
   

# %%
# if __name__ == "__main__":
#     NEAREST_NEIGHBOR = True
    
#     MAX_STEPS = 5000
#     NR_AISLES = 10
#     DEPTH_OF_AISLES = 10
#     warehouse = Warehouse(NR_AISLES, DEPTH_OF_AISLES, 10, 25, max_steps=MAX_STEPS,
#                           generate_pickruns_in_advance=True,
#                           dict_type_state_reward_info=False,
#                           has_congestion=True,
#                           milp_compatible=False,
#                           use_warehouse_coordinates=True,
#                           two_sided_aisles=True,
#                           ensure_diverse_initial_pickruns=True,
#                           pickrun_len_minimum=15,
#                           pickrun_len_maximum=25,
#                           only_current_destinations_available=True,
#                           disruption_freq=50,
#                           disruption_time=60,
#                           use_randomness=True,
#                           use_real_pick_times=True,
#                         #   fixed_seed=0
#                         #   two_sided_aisles=True,
#                         #   has_congestion=True,
#                         #   use_warehouse_coordinates=True)
#     )
#     # warehouse = Warehouse(NR_AISLES, DEPTH_OF_AISLES, 4, 7,
#     #             has_congestion=False,
#     #             generate_pickruns_in_advance=True,
#     #             max_steps=MAX_STEPS,
#     #             dict_type_state_reward_info=False,
#     #             milp_compatible=True,
#     #             ensure_diverse_initial_pickruns=False,
#     #             pickrun_len_minimum=9,
#     #             pickrun_len_maximum=14)
#     warehouse._reset_with_seed(seed=1)
#     warehouse = mo_gym.LinearReward(warehouse, weight=np.array([0.5, 0.5]))
#     scores = []
#     lens = []
#     fairnesses = []
#     times = []
#     workload_values = []
#     for episode in range(1):
#         print(f"Episode {episode} \n")
#         state, info = warehouse.reset(seed=episode+1000) #Initialize state
#         print([amr.pickrun for amr in warehouse.amrs])
#         print(warehouse.current_fairness)
#         # warehouse.max_steps -= len([item for pickrun in warehouse.pickruns for item in pickrun])
#         # warehouse.pickruns = deque([])
#         done = False
#         total_reward = 0
#         i=0
#         while not done:
#             mask = state["mask"]
#             if not np.any(mask != 0):
#                 print(f"{warehouse.steps=},   {mask=}\n {warehouse.pickruns=}")
#             if NEAREST_NEIGHBOR:
#                 action = np.argmin(np.where(mask, state["graph"].nodes[:, 2], np.inf))
#             else:
#                 action = warehouse.action_space.sample(mask=mask)
#             state, reward, done, truncated, info = warehouse.step(action)
#             total_reward += reward
#             if i == MAX_STEPS -1:
#                 done = True
#             if i % 100 == 0:
#                 print(i, warehouse.current_fairness)
#             i+=1
#             # print(f"We tried adding item {action}, this has a weight of {state[0, action]} and added {reward}, leading to total reward so far of {total_reward} and total weight of {state[1, -1]}")
#         fairnesses.append(warehouse.current_fairness)
#         times.append(warehouse.sim.now)
#         print(f"Final reward = {total_reward}, Finishing time = {warehouse.sim.now}, Fairness = {warehouse.current_fairness}, Pick counter = {warehouse.pick_counter}, Jains fairness = {jains_fairness(warehouse.pick_counter)}")
#         print(f"{warehouse.pick_counter=}")
#         scores.append(total_reward)
#         workload_values.append(warehouse.pick_counter.values())
#         lens.append(i)
#     print(f"Average score: {np.mean(scores)}, Std: {np.std(scores)}, CI: {DescrStatsW(scores).tconfint_mean()}")
#     print(f"Average fairness: {np.mean(fairnesses)}, Std: {np.std(fairnesses)}, CI: {DescrStatsW(fairnesses).tconfint_mean()}")
#     print(f"Average time: {np.mean(times)}, Std: {np.std(times)}, CI: {DescrStatsW(times).tconfint_mean()}")
#     print(f"Workload values: {workload_values}")
# %%
# For parallel running using joblib
def run_episode(seed: int):
    NEAREST_NEIGHBOR = True
    
    MAX_STEPS = 5000
    NR_AISLES = 10
    DEPTH_OF_AISLES = 10
    warehouse = Warehouse(NR_AISLES, DEPTH_OF_AISLES, 15, 30, max_steps=MAX_STEPS,
                          generate_pickruns_in_advance=True,
                          dict_type_state_reward_info=False,
                          has_congestion=True,
                          milp_compatible=False,
                          use_warehouse_coordinates=True,
                          two_sided_aisles=True,
                          ensure_diverse_initial_pickruns=True,
                          pickrun_len_minimum=15,
                          pickrun_len_maximum=25,
                          only_current_destinations_available=True,
                          disruption_freq=50,
                          disruption_time=60,
                          use_randomness=True,
                          use_real_pick_times=True,
    )
    warehouse._reset_with_seed(seed=1)
    warehouse = mo_gym.LinearReward(warehouse, weight=np.array([0.5, 0.5]))
    state, info = warehouse.reset(seed=seed) #Initialize state
    print([amr.pickrun for amr in warehouse.amrs])
    print(warehouse.current_fairness)
    done = False
    total_reward = 0
    i=0
    while not done:
        mask = state["mask"]
        if not np.any(mask != 0):
            print(f"{warehouse.steps=},   {mask=}\n {warehouse.pickruns=}")
        if NEAREST_NEIGHBOR:
            action = np.argmin(np.where(mask, state["graph"].nodes[:, 2], np.inf))
        else:
            action = warehouse.action_space.sample(mask=mask)
        state, reward, done, truncated, info = warehouse.step(action)
        total_reward += reward
        if i == MAX_STEPS -1:
            done = True
        if i % 100 == 0:
            print(i, warehouse.current_fairness)
        i+=1
        # print(f"We tried adding item {action}, this has a weight of {state[0, action]} and added {reward}, leading to total reward so far of {total_reward} and total weight of {state[1, -1]}")
    warehouse.save_trajectories(id="GREEDY_10x10_15_30")
    return list({"current_fairness": warehouse.current_fairness,
            "total_reward": total_reward,
            "time": warehouse.sim.now,
            "workload": list(warehouse.pick_counter.values()),}.items())
if __name__=="__main__":
    # pool = mp.Pool(6)
    # results = pool.map(run_episode, range(25))
    # # results = joblib.Parallel(n_jobs=6)(joblib.delayed(run_episode)(seed) for seed in range(6))
    # scores = [dict(result)["total_reward"] for result in results]
    # fairnesses = [dict(result)["current_fairness"] for result in results]
    # times = [dict(result)["time"] for result in results]
    # workload_values = [dict(result)["workload"] for result in results]
    # print(f"Average score: {np.mean(scores)}, Std: {np.std(scores)}, CI: {DescrStatsW(scores).tconfint_mean()}")
    # print(f"Average fairness: {np.mean(fairnesses)}, Std: {np.std(fairnesses)}, CI: {DescrStatsW(fairnesses).tconfint_mean()}")
    # print(f"Average time: {np.mean(times)}, Std: {np.std(times)}, CI: {DescrStatsW(times).tconfint_mean()}")
    # print(f"Workload values: {workload_values}")
    print(run_episode(1))
# %% Vanderlande Benchmark method:

def run_benchmark_episode(seed: int):
    MAX_STEPS = 200
    NR_AISLES = 10
    DEPTH_OF_AISLES = 10
    # warehouse = Warehouse(NR_AISLES, DEPTH_OF_AISLES, 15, 30, max_steps=MAX_STEPS,
    #                             generate_pickruns_in_advance=True,
    #                             dict_type_state_reward_info=False,
    #                             has_congestion=True,
    #                             milp_compatible=False,
    #                             use_warehouse_coordinates=True,
    #                             two_sided_aisles=True,
    #                             ensure_diverse_initial_pickruns=True,
    #                             pickrun_len_minimum=15,
    #                             pickrun_len_maximum=25,
    #                             only_current_destinations_available=True,
    #                             benchmark_warehouse=True,
    #                             fixed_seed=None,
    #                             disruption_freq=50,
    #                             disruption_time=60,
    #                             use_randomness=True,
    #                             use_real_pick_times=True,
    #                             # two_sided_aisles=True,
    #                             #   has_congestion=True,
    #                             #   use_warehouse_coordinates=True)
    #     )
    
    warehouse = Warehouse(7, 7, 4, 7,
                    has_congestion=False,
                    generate_pickruns_in_advance=True,
                    max_steps=200,
                    dict_type_state_reward_info=False,
                    milp_compatible=True,
                    ensure_diverse_initial_pickruns=False,
                    pickrun_len_minimum=9,
                    pickrun_len_maximum=14,
                    benchmark_warehouse=True,)
    warehouse.run_benchmark(seed=seed)
    # warehouse.save_trajectories(id="BENCHMARK_10x10_15_30")
    return list({"current_fairness": np.std(list(warehouse.pick_counter.values())),
                 "time": warehouse.sim.now,
                 "workload": list(warehouse.pick_counter.values()),}.items())

if __name__ == "__main__": 
#     pool = mp.Pool(6)
#     results = pool.map(run_benchmark_episode, range(100))
#     fairnesses = [dict(result)["current_fairness"] for result in results]
#     times = [dict(result)["time"] for result in results]
#     workload_values = [dict(result)["workload"] for result in results]   
#     print("TIME:", np.mean(times), np.std(times), DescrStatsW(times).tconfint_mean())
#     print("FAIRNESS:", np.mean(fairnesses), np.std(fairnesses), DescrStatsW(fairnesses).tconfint_mean())
#     print("WORKLOAD:", workload_values)
    
    print(run_benchmark_episode(1000))
