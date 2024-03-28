# Learning Efficient and Fair Policies for Uncertainty-Aware Collaborative Human-Robot Order Picking
This repo contains the Python code used for the paper 
"Learning Efficient and Fair Policies for Uncertainty-Aware Collaborative Human-Robot Order Picking".

We shortly explain the code files below.

## Folders

* **final_models_single_obj_performance**: Contains the PyTorch models trained for the different single-objective training instances, used in various tests.
* **np_arrays**: Contains the Numpy arrays with the pick quantities and volume/weights info used for the simulation model.
* **pd_dataframes**: Contains Pandas dataframes used in helpers.py.
* **PGMORL**: Contains the multi-objective reinforcement learning code.
* **PGMORL_cluster_results**: Contains the results of the multi-objective reinforcement learning training runs.

## Relevant files

* **env_mo.py**: Contains the simulation environment. At the bottom, code pieces can be commented/uncommented to run the GREEDY or VI BENCHMARK policies.
* **helpers.py**: Contains helper functions for the simulation models, such as creating warehouse graphs, sampling products, etc.
* **model.py**: Contains all relevant PyTorch neural network models.
* **numba_helpers.py**: Contains helper function for the simulation model that was sped up using Numba.
* **ppo_evaluate_run_mo_env.py**: Contains the code to evaluate the trained single-objective PPO models on the simulation environment.
* **ppo_training_run_mo_env.py**: Contains the code to train the single-objective PPO models on the simulation environment.

## Explanation on Code
### Simulation
`env_mo.py` contains the simulation environment. At the bottom, code pieces can be commented/uncommented to run the GREEDY or VI BENCHMARK policies, with options to run them in parallel. For the greedy and DRL methods, the gym interface is used while for the VI Benchmark a separate function is utilized. The simulation consists of 3 main classes.

1. Warehouse: Contains the general warehouse info, such as product locations, distance matrices, and the main Simpy process.
2. Picker: Contains the logic of a picker within the simulation environment.
3. AMR: Contains the logic of an AMR within the simulation environment.

### Single-objective DRL Training
To train the PPO policies, the file `ppo_training_run_mo_env.py` can be used. This file contains the code to train the PPO models on the simulation environment. Add the top of this file, you can specify the warehouse parameters as well as the learning parameters. One option here is the architecture option, with which you can select which network architecture is used. The details of the networks can be found in the model.py file. 

### Single-objective DRL Evaluation
To evaluate the PPO policies, the file `ppo_evaluate_run_mo_env.py` can be used. This file contains the code to evaluate the trained PPO models on the simulation environment. At the top of this file, you can specify the warehouse parameters as well as the learning parameters. 

### Multi-objective DRL Training
Multi-objective DRL training can be found in the `PGMORL` folder. The main relevant training code can be found in the `morl` subdirectory. Here, in reference_point.py, you must specify the reference point to be used in the form [efficiency reference, fairness reference]. In `arguments.py`, the relevant PGMORL parameters can be defined. The PPO parameters in this file are not used. In `run.py`, you can specify the warehouse parameters (NOTE: the objective weight parameters are not used). The train_args dict arguments related to PPO are also not used, but the dimension sizes of the network are. PGMORL uses the objective values that are outputted in the info["obj"] from the state in the simulation model. In this `run.py` file you can specify the PPO parameters.

### Multi-objective DRL Evaluation
To evaluate multi-objective policies, you can use the file `test_evaluation.py`. Here, you can again specify the warehouse parameters. At the bottom of this file, you can specify which policy you want to use and in the evaluate() function you can specify the number of episodes that you want.

## Explanation of Pytorch models
### Single-objective DRL
The saved Pytorch models use a general naming convention that has been adapted a couple of times. The most relevant information, however, can be easily read to know on which warehouse instances the models were trained and with which network architecture. 

The start has some description that is not structured. Then, the network architecture that was used is incorporated. Here, more will have "invariant_mlp_critic_embedding_actor", which is the main proposed aisle-embedding structure. Others are gcn, gin or invariant_mlp. The warehouse parameters are described, having the number of aisles in the warehouse "nr_ai", the depth of each aisle "nr_ra", the number of pickers "nr_pi", and the number of AMRs "nr_am". Then some weight information regarding the objectives is added. Here, "w_per" refers to the weight of the performance reward, "w_fai" to the weight of the fairness reward. Lastly, some models have an extra description regarding the used features. I.e., "fair", "effic", or "both". 

For example, to retrieve the main single-objective efficiency models for the largest two warehouse sizes, we have the following models:

- `good_lar-invariant_mlp_critic_embedding_actor-nr_ai_35_nr_ra_40_nr_pi_60_nr_am_180_w_per_40_w_fai_0_feat__effic.pt`
- `ppo_result_arch-invariant_mlp_critic_embedding_actor-nr_ai_25_nr_ra_25_nr_pi_30_nr_am_90_max_s_7500.pt`

### Multi-objective DRL
The Pytorch models of multi-objective training can be easily found in the folder `PGMORL_cluster_results`. Here you can go into the folder with the name `trained_models_{nr_aisle}_{aisle_depth}_{nr_pickers}_{nr_amrs}`. Then, in the numbered subfolders, intermediate models can be found, and the final outputted models can be found in the `final` folder. Here, each model is the model for a unique policy that focuses on a unique trade-off between efficiency and fairness.