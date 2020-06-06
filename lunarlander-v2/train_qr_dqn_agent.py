import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

import numpy as np
import pandas as pd
import tensorflow as tf

import gym

import os
import sys
import time
import pickle
import random

path = os.path.abspath('..')
if path not in sys.path:
    sys.path.append(path)
from sale.agents.default_config import DEFAULT_CONFIG as config
from sale.agents.qr_dqn import QuantileAgent

SEED = 123456789
np.random.seed(SEED)
tf.random.set_seed(SEED)

stime = time.time()

##### update config for QR-DQN agent #####
config['hiddens'] = [256, 256]
config['max_training_steps'] = 500000
config['lr'] = 5e-4
config['decay_steps'] = 1000000
config['episode_counts_to_save'] = 100
config['persistent_directory'] = 'online/'
config['checkpoint_path'] = 'online/ckpts/'
##### train the agent #####
agent = QuantileAgent(name='LunarLander-v2', num_actions=4, config=config)
agent.learn()

##### training curve of the online agent #####
rewards = pd.Series(agent.eval_episode_rewards)
steps = pd.Series(agent.eval_episode_steps)
fig, axes = plt.subplots(2, 2, figsize=(18, 8))
axes[0][0].plot(rewards.rolling(100, min_periods=20).mean())
axes[0][0].set_title('mean reward')
axes[0][1].plot(rewards.rolling(100, min_periods=20).max())
axes[0][1].set_title('max reward')
axes[1][0].plot(steps.rolling(100, min_periods=20).mean())
axes[1][0].set_title('mean step')
axes[1][1].plot(steps.rolling(100, min_periods=20).max())
axes[1][1].set_title('max step')
plt.savefig('online/figs/qr_dqn_online.png', dpi=200, bbox_inches='tight')

etime = time.time()

print( 'time: {} minutes!'.format((etime-stime)//60) )

##### gather all trajectories #####
files = os.listdir('online')
files = sorted([file for file in files if file.endswith('.pkl')])
trajs = []
for file in files:
    path = 'online/' + file
    with open(path, 'rb') as f:
        trajs.append(pickle.load(f))
trajs = [traj for file in trajs for traj in file]
with open('online/trajs_qr_dqn.pkl', 'wb') as f:
    pickle.dump(trajs, f)
    
for file in files:
    path = 'online/' + file
    os.remove(path)

# ##### offline qr-dqn agent training #####
# from shutil import copy
# copy('online/trajs_qr_dqn.pkl', 'offline/agent/')

# config['online'] = False
# config['max_training_steps'] = 500000
# config['lr'] = 5e-4
# config['decay_steps'] = 1000000
# config['persistent_directory'] = 'offline/agent/'
# config['checkpoint_path'] = 'offline/agent/ckpts/'
# agent = QuantileAgent(name='LunarLander-v2', num_actions=4, config=config)
# agent.learn()

# ##### training curve of the offline agent #####
# rewards = pd.Series(agent.eval_episode_rewards)
# steps = pd.Series(agent.eval_episode_steps)
# fig, axes = plt.subplots(2, 2, figsize=(18, 8))
# axes[0][0].plot(rewards.rolling(100, min_periods=20).mean())
# axes[0][0].set_title('mean reward')
# axes[0][1].plot(rewards.rolling(100, min_periods=20).max())
# axes[0][1].set_title('max reward')
# axes[1][0].plot(steps.rolling(100, min_periods=20).mean())
# axes[1][0].set_title('mean step')
# axes[1][1].plot(steps.rolling(100, min_periods=20).max())
# axes[1][1].set_title('max step')
# plt.savefig('offline/figs/qr_dqn_offline.png', dpi=200, bbox_inches='tight')