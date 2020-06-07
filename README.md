# A Supervised Learning Framework for Batch Reinforcement Learning
This repository is the official implementation of the paper `A Supervised Learning Framework for Batch Reinforcement Learning` submitted to NeurIPS 2020.

## Requirements
- Python version: Python 3.6.8 :: Anaconda custom (64-bit)
### Main packages for the proposed estimator
- numpy == 1.18.1
- pandas == 1.0.3
- sklearn == 0.22.1
- tensorflow == 2.1.0
### Additional packages for experiments
- pickle
- os
- sys
- random

## SALE package Overview
- `models`: network structures
- `agents`: DQN, MultiHeadDQN, QR-DQN agents
- `replay_buffers`: basic and priporitized replay buffers
- `algos`: behavior cloning, density estimator, advantage learner, fitted Q evaluation, etc

## Reproduce simulation results
### Synthetic data
- ```python train_qr_dqn_agent.py &``` in the lunarlander-v2 folder: online training a QR-DQN agent in the Gym LunarLander-v2 enviroment, this takes nearly three hours without GPU support. 
- copy the `trajs_qr_dqn.pkl` under `online` folder produced by the first step to  `dqn_2_200/random/` folder, and run ```python batch_sale_random_dqn.py &``` (around 20 hours without GPU support). This will generate DQN offline training results. Similarly, we can obtain DDQN, QR-DQN results, when we use random or the first 200 trajectories, our results are given in `lunarlander-v2/plot_figs`.
-  ```python plot_ckpts_avg_figs.py & ``` and ```python plot_ckpts_last_figs.py &``` to generate figures in our paper.
### Real data based simulation
- due to the data confidentiality agreement, we cannot provide the real datasets. However, we do provide simulated data that minic the real dataset
- run the scripts under realdata after putting `trajs.pkl` of real data in the `realdata/data` folder. `trajs.pkl` are a list of list of transitions `(s,a,r,s',done)`

### Computational complexity
we assume that the forward and backward of network complexity is `S`
- step 2: training `L` DQN agents, batch size `B_1`, training steps `I_1`, total `O(L * I_1 * B_1 * S)`
- step 3: training `L` density estimators, batch size `B_2`, training steps `I_2`, total `O(L * I_2 * B_2^4 * S)`
- step 4: pseudo Q computations, batch size `B_3`, total `O(B_3 * N * T * A * S)`, where `N` number of trajs, `T` average length of trajs, `A` number of actions.
- step 5: training tau, batch size `B_4`, training steps `I_4`, total `O(I_4 * B_4 * S)`

#### Thanks to Repos:
- https://github.com/google-research/batch_rl
- https://github.com/ray-project/ray
