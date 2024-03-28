import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

from copy import deepcopy

import gymnasium as gym
import torch
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.envs import make_env, make_vec_envs
from a2c_ppo_acktr.model import Policy
from mopg import evaluate_parallel, evaluation
from sample import Sample
from scalarization_methods import WeightedSumScalarization
from tqdm import tqdm
from utils import generate_weights_batch_dfs

train_args = {"hidden_dim_actor": 64, #32 #64
            "output_dim_actor": 16, #8 #16 
            "hidden_dim_critic": 64, #32 $64
            "output_dim_critic": 16, #8 # 16
            "lr":5e-4, #5e-3 for regular training
            "discount_factor":0.995,
            "batch_size":128,
            "max_batch_size_ppo": 0, #3200, 3600 # 0 means step_per_collect amount
            "nr_envs": 2, # 64
            "max_epoch": 100,
            "step_per_epoch": 800, #6400, 25600, 12800
            "repeat_per_collect": 3}

'''
initialize_warm_up_batch: method to generate tasks in the warm-up stage.
Each task is a pair of an initial random policy and an evenly distributed optimization weight.
The optimization weight is represented by a weighted-sum scalarization function.
'''
def initialize_warm_up_batch(args, device):
    # using evenly distributed weights for warm-up stage
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    sample_batch = []
    scalarization_batch = []

    temp_env = gym.make('warehouse', disable_env_checker=True) # temp_env is only used for initialization

    counter = 0
    for weights in tqdm(weights_batch):
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
            obj_num=args.obj_num)
        if device == 'cuda':
            device = f"cuda:{counter % torch.cuda.device_count()}"
        actor_critic.to(device).double()

        if args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=1e-5,
                max_grad_norm=args.max_grad_norm)
        else:
            # NOTE: other algorithms are not supported yet
            raise NotImplementedError
    
        envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                            gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                            obj_rms = args.obj_rms, ob_rms = args.ob_rms)
        envs.ob_rms = None; envs.ret_rms = None; envs.obj_rms = None
        env_params = {}
        env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
        env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
        env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None
        envs.close()

        scalarization = WeightedSumScalarization(num_objs = args.obj_num, weights = weights)

        sample = Sample(env_params, actor_critic, agent, optgraph_id = -1)

        objs = evaluate_parallel(actor_critic=sample.actor_critic, ob_rms=None,
                                 env_name=args.env_name, seed=args.seed,
                                 num_processes=20, eval_log_dir=None, device=device,
                                 nr_eval_episodes=args.eval_num)
        sample.objs = objs

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)
        
        counter +=1
    
    temp_env.close()

    return sample_batch, scalarization_batch