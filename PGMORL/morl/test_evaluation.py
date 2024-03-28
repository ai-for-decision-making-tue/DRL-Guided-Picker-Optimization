import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))



from copy import deepcopy

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_env, make_vec_envs
from a2c_ppo_acktr.model import Policy
from mopg import evaluation
from sample import Sample
from scalarization_methods import WeightedSumScalarization
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm
from utils import generate_weights_batch_dfs

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env_mo import Warehouse

env_args = {    "nr_aisles": 10,
                "nr_racks_per_aisle": 10,
                "nr_pickers": 10,
                "nr_amrs": 25,
                "max_steps": 5000, # 5000
                "generate_pickruns_in_advance": True,
                "has_congestion": True,
                "milp_compatible": False,
                "only_current_destinations_available": False,
                "diverse_starting": True,
                "pickrun_len_minimum": 15, # 15
                "pickrun_len_maximum": 25, # 25
                "disruption_freq": 50,
                "disruption_time": 60,
                "use_randomness": True,
                "use_real_pick_times": True,
                "w_perf": 0.1, 
                "w_fair": 1,}
train_args = {"hidden_dim_actor": 64, #32 #64
            "output_dim_actor": 16, #8 #16 
            "hidden_dim_critic": 64, #32 $64
            "output_dim_critic": 16, #8 # 16
            "lr":5e-4,
            "discount_factor":0.995,
            "batch_size":128,
            "max_batch_size_ppo": 0, #3200, 3600 # 0 means step_per_collect amount
            "nr_envs": 2, # 64
            "max_epoch": 100,
            "step_per_epoch": 800, #6400, 25600, 12800
            "repeat_per_collect": 3}

def return_warehouse_instance(seed=None):
    warehouse = Warehouse(env_args["nr_aisles"], env_args["nr_racks_per_aisle"],
                env_args["nr_pickers"], env_args["nr_amrs"],
                dict_type_state_reward_info=False,
                max_steps=env_args["max_steps"],
                generate_pickruns_in_advance=env_args["generate_pickruns_in_advance"],
                has_congestion=env_args["has_congestion"],
                milp_compatible=env_args["milp_compatible"],
                only_current_destinations_available=env_args["only_current_destinations_available"],
                ensure_diverse_initial_pickruns=env_args["diverse_starting"],
                pickrun_len_maximum=env_args["pickrun_len_maximum"],
                pickrun_len_minimum=env_args["pickrun_len_minimum"],
                disruption_freq=env_args["disruption_freq"],
                disruption_time=env_args["disruption_time"],
                use_randomness=env_args["use_randomness"],
                use_real_pick_times=env_args["use_real_pick_times"],
                fixed_seed=None)
    warehouse.reset(seed=seed)
    warehouse = mo_gym.LinearReward(warehouse, weight=np.array([0.1, 1]))
    return warehouse
def warehouse_function_returner(seed):
    return return_warehouse_instance

gym.register("warehouse", warehouse_function_returner(0))

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    
    eval_episode_efficiency_scores = []
    eval_episode_fairness_scores = []
    final_times = []
    final_fairnesses = []
    final_workload_values = []
    
    
    efficiency_scores = [0 for i in range(eval_envs.num_envs)]
    fairness_scores = [0 for i in range(eval_envs.num_envs)]

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 100:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for i, info in enumerate(infos):
            efficiency_score = info["obj"][0]
            fairness_score = info["obj"][1]
            efficiency_scores[i] += efficiency_score
            fairness_scores[i] += fairness_score
        
        
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                eval_episode_efficiency_scores.append(efficiency_scores[i])
                eval_episode_fairness_scores.append(fairness_scores[i])
                final_times.append(info["now"])
                final_fairnesses.append(info["current_fairness"])
                final_workload_values.append(info["workload_values"])
                efficiency_scores[i] = 0
                fairness_scores[i] = 0

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f} ({})\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards),
        DescrStatsW(eval_episode_rewards).tconfint_mean()))
    print(" Evaluation using {} episodes: mean efficiency {:.5f} ({})\n".format(
        len(eval_episode_efficiency_scores), np.mean(eval_episode_efficiency_scores),
        DescrStatsW(eval_episode_efficiency_scores).tconfint_mean()))
    print(" Evaluation using {} episodes: mean fairness {:.5f} ({})\n".format(
        len(eval_episode_fairness_scores), np.mean(eval_episode_fairness_scores),
        DescrStatsW(eval_episode_fairness_scores).tconfint_mean()))
    print(" Evaluation using {} episodes: mean final time {:.5f} ({})\n".format(
        len(final_times), np.mean(final_times),
        DescrStatsW(final_times).tconfint_mean()))
    print(" Evaluation using {} episodes: mean final fairness {:.5f} ({})\n".format(
        len(final_fairnesses), np.mean(final_fairnesses),
        DescrStatsW(final_fairnesses).tconfint_mean()))
    print("Final workload values:", final_workload_values)

if __name__=="__main__":
    temp_env = gym.make('warehouse', disable_env_checker=True)
    actor_critic = Policy(
                temp_env.observation_space.shape,
                temp_env.action_space,
                base_kwargs={"in_channels_fair": 13,
                            "in_channels_efficient": 23,
                            "hidden_channels": train_args["hidden_dim_actor"],
                            "emb_channels": train_args["output_dim_actor"],
                            "hidden_after_emb": train_args["hidden_dim_actor"],
                            "out_channels": train_args["output_dim_actor"],
                            },
                obj_num=2)
    actor_critic.to(device).double()
    actor_critic.load_state_dict(torch.load(os.path.join("cluster_results",
                                                         "trained_models_10x10_10_25",
                                                         "final", "EP_policy_3.pt"),
                                                         map_location=device))
    evaluate(actor_critic=actor_critic, env_name="warehouse", num_processes=6, device=device,
             ob_rms=None, seed=0, eval_log_dir=None)