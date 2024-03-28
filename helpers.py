# %%
import operator
import os
from typing import Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

DF = pd.read_pickle(os.path.join(
    "pd_dataframes", "anon_product_volume_weight_group.pkl"))
AISLE_PRODUCT_COUNT = pd.read_pickle(os.path.join(
    "pd_dataframes", "anon_aisle_product_count.pkl"))

PICK_QUANTITY_ARRAY = np.load(os.path.join("np_arrays", "pick_quantity_array_2.npy"))
VOLUME_WEIGHT_ARRAY = np.load(os.path.join("np_arrays", "volume_weight_array.npy"))

x = 4
y = 6


def _create_unidirectional_from_grid(graph: nx.DiGraph, x: int, y: int,
                                     has_two_sided_aisles: bool) -> nx.DiGraph:
    if has_two_sided_aisles:
        for aisle in range(x):
            for y_coor in range(1, y):
                for aisle_side in range(2):
                    if aisle % 2 == 0:
                        graph.remove_edge((2 *aisle + aisle_side, y_coor),
                                          (2*aisle + aisle_side, y_coor - 1))
                    else:
                        graph.remove_edge((2 *aisle + aisle_side, y_coor - 1),
                                        (2*aisle + aisle_side, y_coor))
    
    else:
        for x_coor in range(x):
            for y_coor in range(1, y):
                if x_coor % 2 == 0:
                    graph.remove_edge((x_coor, y_coor), (x_coor, y_coor - 1))
                else:
                    graph.remove_edge((x_coor, y_coor - 1), (x_coor, y_coor))
    return graph

def _remove_connections_between_aisles(graph: nx.DiGraph, x: int, y: int,
                                       has_two_sided_aisles: bool) -> nx.DiGraph:
    if has_two_sided_aisles:
        for aisle in range(1, x):
            for y_coor in range(1, y - 1):
                    graph.remove_edge((2 *aisle, y_coor),
                                      (2*aisle - 1, y_coor))
                    graph.remove_edge((2 *aisle - 1, y_coor),
                                      (2*aisle , y_coor))
    else:
        for x_coor in range(1, x):
            for y_coor in range(1, y - 1):
                graph.remove_edge((x_coor, y_coor), (x_coor - 1, y_coor))
                graph.remove_edge((x_coor - 1, y_coor), (x_coor, y_coor))
    return graph

def create_coordinate_dicts(x: int, y: int, step_distance: float = 1.4,
                             cross_distance: float = 1.0,
                             between_aisle_distance: float = 6) -> Tuple[dict, dict]:
    x_coordinate_dict = {}
    y_coordinate_dict = {} 
    for aisle in range(x):
        for aisle_side in range(2):
            for y_coor in range(y):
                x_value = cross_distance * aisle_side + (between_aisle_distance + cross_distance) * aisle
                y_value = step_distance * y_coor
                y_coordinate_dict[(int(aisle*2 + aisle_side), y_coor)] = x_value
                x_coordinate_dict[(int(aisle*2 + aisle_side), y_coor)] = y_value
    return x_coordinate_dict, y_coordinate_dict

def _add_edge_distances(graph: Union[nx.Graph, nx.DiGraph], x: int, y: int,
                           step_distance: float = 1.4,
                           cross_distance: float = 1.0,
                           between_aisle_distance: float = 6):
    x_coordinate_dict, y_coordinate_dict = create_coordinate_dicts(
        x, y, step_distance, cross_distance, between_aisle_distance
    )
    edge_distance_dict = {}
    for edge in graph.edges:
        edge_distance_dict[edge] = max(
            abs(x_coordinate_dict[edge[0]] - x_coordinate_dict[edge[1]]),
            abs(y_coordinate_dict[edge[0]] - y_coordinate_dict[edge[1]])
        )
    nx.set_edge_attributes(graph, edge_distance_dict, name="weight")
    return graph 
    

def create_warehouse_grid(x, y, is_unidirectional: bool, two_sided_aisles: bool,
                          use_warehouse_coordinates: bool):
    if two_sided_aisles:
        graph = nx.grid_2d_graph(int(2*x), y)
        graph = graph.to_directed()
        # Create Unidirectional aisles
        # Each aisle has breadth of 2
        if is_unidirectional:
            graph = _create_unidirectional_from_grid(graph, x, y,
                                                     has_two_sided_aisles=True)
        # Remove connections between aisles, except top row and bottom row
        graph = _remove_connections_between_aisles(graph, x, y,
                                                   has_two_sided_aisles=True)
        if use_warehouse_coordinates:
            graph = _add_edge_distances(graph, x, y)
    else:
        if use_warehouse_coordinates:
            raise ValueError("Warehouse coordinates can only be used with two sided aisles")
        graph = nx.grid_2d_graph(x, y)
        graph = graph.to_directed()
        # Create unidirectional aisles
        if is_unidirectional:
            graph = _create_unidirectional_from_grid(graph, x, y,
                                                     has_two_sided_aisles=False)
        # Remove connections between aisles, except top row and bottom row        
        graph = _remove_connections_between_aisles(graph, x, y,
                                                   has_two_sided_aisles=False)
    return graph


def calculate_expected_pick_time(pickqty, volume, weight):
    """
    NOTE, this formula has been masked, as it was based on confidential
    company data. You can insert your own formula here, or sample from any preferred
    distribution.
    """
    pick_time = np.max(0.5, np.random.normal(12.3, 10.8))
    return pick_time

def sample_quantities(nr_of_samples: int,
                      generator: np.random.Generator,
                      with_replacement: bool = True):
    sampled_quantities = generator.choice(
        PICK_QUANTITY_ARRAY, size=nr_of_samples, replace=with_replacement
    )
    return sampled_quantities

def sample_volume_weights(nr_of_samples: int,
                          generator: np.random.Generator,
                          with_replacement: bool = False):
    sampled_volume_weigths = generator.choice(
        VOLUME_WEIGHT_ARRAY, size=nr_of_samples, replace=with_replacement
    )
    return sampled_volume_weigths
    

def jains_fairness(load_per_picker: dict[int, float]):
    return 1 / (
        1 + np.std(list(load_per_picker.values())) /
        np.mean(list(load_per_picker.values()))
    )


def get_sorted_locations_per_aisle(warehouse) -> dict[int, list[int]]:
    # Initialize dictionary
    locations_per_aisle = {aisle: [] for aisle in range(warehouse.x)}
    # Gather the locations per aisle
    for location in range(warehouse.action_space.n):
        aisle = warehouse.find_aisle_from_location(location)
        coordinate_x = warehouse.coordinates_x[location]
        coordinate_y = warehouse.coordinates_y[location]
        locations_per_aisle[aisle].append(
            (location, coordinate_x, coordinate_y))
    # Sort the locations of each aisle based on their depth and their side
    sorted_locations_with_coordinates_per_aisle = dict(map(
        lambda key_value: (
            key_value[0], sorted(key_value[1], key=operator.itemgetter(1, 2))
        ),
        locations_per_aisle.items()))
    # Remove the coordinates from the locations
    sorted_locations_per_aisle = {
        aisle: [loc_details[0] for loc_details in list_of_locs]
        for aisle, list_of_locs in
        sorted_locations_with_coordinates_per_aisle.items()}
    return sorted_locations_per_aisle


def sample_volume_weights_with_categories(
    warehouse,
    nr_samples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    sorted_locations_per_aisle = get_sorted_locations_per_aisle(warehouse)
    done = False
    volume_weight_array = np.zeros((warehouse.action_space.n, 2))
    while not done:
        sampled_category = generator.choice(DF.ARTGRPNAME.unique())
        nr_items = int(generator.choice(AISLE_PRODUCT_COUNT[sampled_category]))
        found_location = False
        for aisle_id, locs in sorted_locations_per_aisle.items():
            if len(locs) >= nr_items:
                sampled_locs = [locs.pop(0) for _ in range(nr_items)]
                sampled_product_ids = generator.choice(
                    DF[DF.ARTGRPNAME == sampled_category].index, nr_items)
                sampled_products = DF[DF.ARTGRPNAME == sampled_category][[
                    "VOLUME", "WEIGHT"]].loc[sampled_product_ids].to_numpy()
                volume_weight_array[sampled_locs] = sampled_products
                found_location = True
                break
        if not found_location:
            for aisle_id, locs in sorted_locations_per_aisle.items():
                if len(locs) > 0:
                    sampled_locs = [locs.pop() for _ in range(len(locs))]
                    sampled_product_ids = generator.choice(
                        DF[DF.ARTGRPNAME == sampled_category].index, len(sampled_locs))
                    sampled_products = DF[DF.ARTGRPNAME == sampled_category][[
                        "VOLUME", "WEIGHT"]].loc[sampled_product_ids].to_numpy()
                    volume_weight_array[sampled_locs] = sampled_products
                    found_location = True
                    break
        if not found_location:
            break
    return volume_weight_array
    

if __name__=='__main__':
    # Plot the grid graph with its edges
    import matplotlib.pyplot as plt

    G = create_warehouse_grid(x, y, True, True, True)
    pos = {node: node for node in G.nodes()}
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()
    # %%
