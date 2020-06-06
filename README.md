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
- models: network structures
- agents: DQN, MultiHeadDQN, QR-DQN agents
- replay_buffers: basic and priporitized replay buffers
- algos: behavior cloning, density estimator, advantage learner, fitted Q evaluation, etc

## Reproduce simulation results
### Synthetic data
- ```python train_qr_dqn_agent.py &``` in the lunarlander-v2 folder: online training a QR-DQN agent in the Gym LunarLander-v2 enviroment, this takes nearly three hours without GPU support. 
- Copy the `trajs_qr_dqn.pkl` under `online` folder produced by the first step to  `dqn_2_200/random/` folder, and run ```python batch_sale_random_dqn.py &``` (around 20 hours without GPU support). This will generate DQN offline training results. Similarly, we can obtain DDQN, QR-DQN results, when we use random or the first 200 trajectories, our results are given in `lunarlander-v2/plot_figs`.
-  ```python plot_ckpts_avg_figs.py & ``` and ```python plot_ckpts_last_figs.py &``` to generate figures in our paper.
### Real data based simulation
- separately run the script under realdata after putting `trajs.pkl` of real data in the `realdata/data` folder. `trajs.pkl` are a list of list of transitions `(s,a,r,s',done)`
