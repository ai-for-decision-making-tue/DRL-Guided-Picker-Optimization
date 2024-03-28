import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines/'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail/'))

import dask
import gymnasium as gym
import mo_gymnasium as mo_gym
import morl
import numpy as np
import torch
from arguments import get_parser
from dask.distributed import Client

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
dask.config.set({"distributed.worker.daemon": False})

gym.logger.set_level(40)
np.seterr(invalid="ignore")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env_mo import Warehouse

env_args = {    "nr_aisles": 4,
                "nr_racks_per_aisle": 4,
                "nr_pickers": 3,
                "nr_amrs": 5,
                "max_steps": 100, # 5000
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

class Logger:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.logfile.write(data)

    def flush(self):
        pass

# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list):
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)

def main(dask_client):
    torch.set_default_dtype(torch.float64)
    
    # ppo parameters
    args_list = ['--lr', '5e-4',
                #  '--use-linear-lr-decay', 'False',
                 '--gamma', '0.995',
                 '--use-gae',
                 '--gae-lambda', '0.95',
                 '--entropy-coef', '0.01',
                 '--value-loss-coef', '0.5',
                 '--num-steps', '400', # 2048
                 '--num-processes', '4',
                 '--ppo-epoch', '3', # 10
                 '--num-mini-batch', '20', # 32
                 '--use-proper-time-limits',
                 '--ob-rms',
                 '--obj-rms',
                 '--raw']

    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    # build saving folder
    save_dir = args.save_dir
    try:
        os.makedirs(save_dir, exist_ok = True)
    except OSError:
        pass
    
    # output arguments
    fp = open(os.path.join(save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()

    logfile = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    sys.stdout = Logger(sys.stdout, logfile)

    morl.run(args, client=dask_client)

    logfile.close()

if __name__ == "__main__":
    dask_client = Client()
    main(dask_client=dask_client)
