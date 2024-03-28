import os
import pickle
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np
import tianshou as ts
import torch
import torch_geometric
from gymnasium.spaces.graph import GraphInstance
from statsmodels.stats.weightstats import DescrStatsW
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from env_mo import Warehouse
from model import (
    GCNNetTianshouPPO_ACTOR, GCNNetTianshouPPO_CRITIC, GINNetTianshouPPO_ACTOR,
    GINNetTianshouPPO_CRITIC, InvariantMLP_WITH_EMBEDDINGTianshouPPO_CRITIC,
    InvariantMLP_WITH_EMBEDDINNGTianshouPPO_ACTOR,
    InvariantMLPTianshou_PER_CLASS_PPO_ACTOR,
    InvariantMLPTianshou_PER_CLASS_PPO_CRITIC, InvariantMLPTianshouPPO_ACTOR,
    InvariantMLPTianshouPPO_CRITIC,
    invariantMLP_WITH_EMBEDDING_PER_CLASS_TIANSHOU_ACTOR)

env_args = {"nr_aisles": 25,
            "nr_racks_per_aisle": 25,
            "nr_pickers": 30,
            "nr_amrs": 70,
            "max_steps": 7500, # 5000
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
            "w_perf": 40, # reward weight efficiency
            "w_fair": 0, # reward weight fairness
            "feat_category": "effic"} # effic, fair, both
# Architecture options are the following:
# - "invariant_mlp"
# - "invariant_mlp_with_embedding",
# - "gcn"
# - "gin"
# - "invariant_mlp_critic_embedding_actor"
# - "invariant_mlp_per_class"
# - "invariant_mlp_critic_embedding_actor_per_class"
train_args = {"architecture": "invariant_mlp_critic_embedding_actor",
                "hidden_dim_actor": 64,
                "output_dim_actor": 16,
                "hidden_dim_critic": 64,
                "output_dim_critic": 16,
                "lr": 5e-4,
                "discount_factor": 0.995,
                "batch_size": 128,
                "max_batch_size_ppo": 0,
                "nr_envs": 64,
                "max_epoch": 150,
                "step_per_epoch": 25600,
                "repeat_per_collect": 3}


env = Warehouse(env_args["nr_aisles"], env_args["nr_racks_per_aisle"],
                env_args["nr_pickers"], env_args["nr_amrs"],
                dict_type_state_reward_info=False,
                max_steps=env_args["max_steps"],
                has_congestion=env_args["has_congestion"],
                milp_compatible=env_args["milp_compatible"],
                only_current_destinations_available=env_args["only_current_destinations_available"],
                pickrun_len_minimum=env_args["pickrun_len_minimum"],
                pickrun_len_maximum=env_args["pickrun_len_maximum"],
                disruption_freq=env_args["disruption_freq"],
                disruption_time=env_args["disruption_time"],
                use_randomness=env_args["use_randomness"],
                use_real_pick_times=env_args["use_real_pick_times"],
                feat_category=env_args["feat_category"],)

print(f"env_args: {env_args} \n train_args: {train_args}")
device = "cuda" if torch.cuda.is_available() else "cpu"

def warehouse_function_returner(seed):
    def return_warehouse_instance(seed=seed):
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
                    feat_category=env_args["feat_category"],
                    fixed_seed=None)
        warehouse = mo_gym.LinearReward(
                warehouse, weight=np.array([env_args["w_perf"], env_args["w_fair"]]))
        warehouse.reset(seed=None)
        return warehouse
    return return_warehouse_instance
if __name__ == "__main__":
    test_envs = ts.env.SubprocVectorEnv(
        [warehouse_function_returner(seed=seed)
        for seed in range(6)]
    )

    if train_args["architecture"] == "invariant_mlp":
        actor_net = InvariantMLPTianshouPPO_ACTOR(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_actor"],
            train_args["output_dim_actor"]
        ).to(device).share_memory()
        critic_net = InvariantMLPTianshouPPO_CRITIC(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_critic"],
            train_args["output_dim_critic"]
        ).to(device).share_memory()
    elif train_args["architecture"] == "invariant_mlp_with_embedding":
        actor_net = InvariantMLP_WITH_EMBEDDINNGTianshouPPO_ACTOR(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_actor"],
            train_args["output_dim_actor"],
            hidden_after_emb=train_args["hidden_dim_actor"],
            out_channels=train_args["output_dim_actor"]
        ).to(device)
        critic_net = InvariantMLP_WITH_EMBEDDINGTianshouPPO_CRITIC(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_critic"],
            train_args["output_dim_critic"],
            hidden_after_emb=train_args["hidden_dim_critic"],
            out_channels=train_args["output_dim_critic"]
        ).to(device)
    elif train_args["architecture"] == "gin":
        actor_net = GINNetTianshouPPO_ACTOR(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_actor"],
            train_args["output_dim_actor"],
        ).to(device)
        critic_net = GINNetTianshouPPO_CRITIC(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_critic"],
            train_args["output_dim_critic"],
        ).to(device)
    elif train_args["architecture"] == "gcn":
        actor_net = GCNNetTianshouPPO_ACTOR(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_actor"],
            train_args["output_dim_actor"],
        ).to(device)
        critic_net = GCNNetTianshouPPO_CRITIC(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_critic"],
            train_args["output_dim_critic"],
        ).to(device)
    elif train_args["architecture"] == "invariant_mlp_critic_embedding_actor":
        actor_net = InvariantMLP_WITH_EMBEDDINNGTianshouPPO_ACTOR(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_actor"],
            train_args["output_dim_actor"],
            hidden_after_emb=train_args["hidden_dim_actor"],
            out_channels=train_args["output_dim_actor"]
        ).to(device)
        critic_net = InvariantMLPTianshouPPO_CRITIC(
            env.observation_space["graph"].node_space.shape[0],
            train_args["hidden_dim_critic"],
            train_args["output_dim_critic"]
        ).to(device).share_memory()
    elif train_args["architecture"] == "invariant_mlp_critic_embedding_actor_per_class":
        actor_net = invariantMLP_WITH_EMBEDDING_PER_CLASS_TIANSHOU_ACTOR(
            13, 23, train_args["hidden_dim_actor"],
            train_args["output_dim_actor"],
            hidden_after_emb=train_args["hidden_dim_actor"],
            out_channels=train_args["output_dim_actor"]
        ).to(device)
        critic_net = InvariantMLPTianshou_PER_CLASS_PPO_CRITIC(
            13, 23, train_args["hidden_dim_critic"],
            train_args["output_dim_critic"],
        ).to(device)
    elif train_args["architecture"] == "invariant_mlp_per_class":
        actor_net = InvariantMLPTianshou_PER_CLASS_PPO_ACTOR(
            13, 23, train_args["hidden_dim_actor"], train_args["output_dim_actor"],
        ).to(device)
        critic_net = InvariantMLPTianshou_PER_CLASS_PPO_CRITIC(
            13, 23, train_args["hidden_dim_critic"], train_args["output_dim_critic"],
        ).to(device)
    else:
        raise NotImplementedError("The selected architecture is not implemented")

    
    optim = torch.optim.Adam(
        params=list(actor_net.parameters()) + list(critic_net.parameters()), 
        lr=train_args["lr"]
    )

    policy = ts.policy.PPOPolicy(actor_net, critic_net, optim,
                                discount_factor=train_args["discount_factor"],
                                dist_fn=torch.distributions.categorical.Categorical,
                                deterministic_eval=True
                                )
    policy.action_type = "discrete"

    def preprocess_function(**kwargs):
        if "obs" in kwargs:
            obs_with_tensors = [
                {"graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
                "graph_edges": torch.from_numpy(obs["graph"].edges).float(),
                "graph_edge_links": torch.from_numpy(obs["graph"].edge_links.T).int(),
                "mask":torch.from_numpy(obs["mask"]).to(torch.int8),
                "picks_left":torch.tensor(obs["picks_left"], dtype=torch.int32),
                "aisle_nrs": torch.from_numpy(obs["Aisle_nrs"]).to(torch.int16)}
                for obs in kwargs["obs"]]
            kwargs["obs"] = obs_with_tensors
        if "obs_next" in kwargs:
            obs_with_tensors = [
                {"graph_nodes": torch.from_numpy(obs["graph"][0]).float(),
                "graph_edges": torch.from_numpy(obs["graph"][1]).float(),
                "graph_edge_links": torch.from_numpy(obs["graph"][2].T).int(),
                "mask":torch.from_numpy(obs["mask"]).to(torch.int8),
                "picks_left":torch.tensor(obs["picks_left"], dtype=torch.int32),
                "aisle_nrs": torch.from_numpy(obs["Aisle_nrs"]).to(torch.int16)}
                for obs in kwargs["obs_next"]]
            kwargs["obs_next"] = obs_with_tensors
        return kwargs

    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True, preprocess_fn=preprocess_function)
    test_collector.reset()
    

    policy.eval()
    collector = ts.data.Collector(policy, test_envs, exploration_noise=False, preprocess_fn=preprocess_function)
    result = collector.collect(n_episode=100)
    reward_histories = test_envs.get_env_attr("reward_history")
    fairness_results = [fairness for history in reward_histories for fairness in history["fairness"]]
    pick_time_results = [pick_time for history in reward_histories for pick_time in history["time"]]
    workload_values = [workload for history in reward_histories for workload in history["workload_values"]]
    print(result)
    print(f"Average reward over {len(result['rews'])} episodes: {result['rew']}, std: {np.std(result['rews'])}, CI: {DescrStatsW(result['rews']).tconfint_mean()} \n Avg Fairness = {np.mean(fairness_results)}, std: {np.std(fairness_results)}, CI: {DescrStatsW(fairness_results).tconfint_mean()} \n Avg Pick Time = {np.mean(pick_time_results)}, std: {np.std(pick_time_results)}, CI: {DescrStatsW(pick_time_results).tconfint_mean()}")
    print(f"Workload values: {workload_values}")