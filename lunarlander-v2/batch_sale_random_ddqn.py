from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import random

import os
import sys
import copy
import pickle

path = os.path.abspath('..')
if path not in sys.path:
    sys.path.append(path)

from sale.agents.default_config import DEFAULT_CONFIG as config
from sale.agents.dqn import DQNAgent
from sale.agents.qr_dqn import QuantileAgent

from sale.algos.kfold import KFoldCV

from sale.algos.advantage_learner import AdvantageLearner
from sale.algos.behavior_cloning import BehaviorCloning
from sale.algos.density_ratio import VisitationRatioModel

from collections import defaultdict

def set_config(kf, lr=5e-4, decay_step=int(1e4), max_train_step=int(5e4)):
    config['online'] = False
    config['hiddens'] = [256, 256]
    config['lr'] = lr
    config['decay_steps'] = decay_step
    config['max_training_steps'] = max_train_step
    config['persistent_directory'] = kf.agent_path
    config['checkpoint_path'] = kf.ckpt_path
    config['training_steps_to_checkpoint'] = 10000

    config['dueling'] = False
    config['double'] = True


def compare_within_ckpt(kf, bc, config, working_directory,
                        strategy = 'random',
                        num_trajectories = 200,
                        agent_name = 'dqn',
                        num_kf =2,
                        replica=1):
    # 0 for sale, 1 for dml, 2 for single agent
    ckpt_result = defaultdict(list)

    for ckpt in [i * int(1e4) for i in range(1, int(config['max_training_steps'] / 1e4) + 1)]:
        print('Evaluate with ckpt {}...'.format(ckpt))
        agents = []

        for idx in range(kf.n_splits):
            config_idx = copy.deepcopy(config)
            config_idx['persistent_directory'] = kf.agent_paths[idx]
            config_idx['checkpoint_path'] = kf.ckpt_paths[idx]

            agent_idx = DQNAgent(name='LunarLander-v2', num_actions=4, config=config_idx)
            agent_idx.load(kf.ckpt_paths[idx] + 'dqn_{}.ckpt'.format(ckpt))
            agents.append(agent_idx)

        states, qvalues, qtildes = kf.update_q(agents, bc)

        advs1 = qvalues - qvalues.mean(axis=1, keepdims=True)
        adv_learner1 = AdvantageLearner()
        adv_learner1._train(states, advs1)
        adv_learner1._eval(100)

        advs2 = qtildes - qtildes.mean(axis=1, keepdims=True)
        adv_learner2 = AdvantageLearner()
        adv_learner2._train(states, advs2)
        adv_learner2._eval(100)

        eval_episode_rewards1 = np.array(adv_learner1.eval_episode_rewards)
        eval_episode_rewards2 = np.array(adv_learner2.eval_episode_rewards)

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        axes[0].hist(eval_episode_rewards1)
        axes[0].set_title(eval_episode_rewards1.mean())
        axes[1].hist(eval_episode_rewards2)
        axes[1].set_title(eval_episode_rewards2.mean())

        ad_pic_file_path = os.path.join(
            working_directory, 'pic', 'ag_{}-sp_{}-nt_{}-kf_{}-ckpt_{}-dt_{}_adv.jpg'.format(
            agent_name, strategy, num_trajectories, num_kf, ckpt, datetime.now().strftime('%Y%m%d_%H-%M-%s')))
        plt.savefig(ad_pic_file_path)

        # record dml, sale rewards
        ckpt_result['dml_mean_reward'].append(eval_episode_rewards1.mean())
        ckpt_result['sale_mean_reward'].append(eval_episode_rewards2.mean())

        [agents[idx]._eval(100) for idx in range(kf.n_splits)]

        # record cv agent rewards
        print('Evaluating cv agent score...')
        cv_agent_rewards = [np.array(agents[idx].eval_episode_rewards).mean() for idx in range(kf.n_splits)]
        for idx, ar in enumerate(cv_agent_rewards):
            ckpt_result['cv{}'.format(idx)].append(ar)
        print(cv_agent_rewards)

        print('Evaluating single score...')
        config['persistent_directory'] = kf.agent_path
        config['checkpoint_path'] = kf.ckpt_path

        agent = DQNAgent(name='LunarLander-v2', num_actions=4, config=config)
        agent.load(config['checkpoint_path']+'dqn_{}.ckpt'.format(ckpt))

        agent._eval(100)

        eval_episode_rewards = np.array(agent.eval_episode_rewards)
        ckpt_result['single_agent_mean_reward'].append(eval_episode_rewards.mean())

        plt.hist(eval_episode_rewards)
        plt.title(eval_episode_rewards.mean())

        single_pic_file_path = os.path.join(
            working_directory, 'pic',
            'ag_{}-sp_{}-nt_{}-kf_{}-ckpt_{}-dt_{}_single.jpg'.format(agent_name, strategy, num_trajectories, num_kf,
                                                                        ckpt, datetime.now().strftime('%Y%m%d_%H-%M-%s')))
        plt.savefig(single_pic_file_path)

        print('Recording check point results...')

    ckpt_result_pdf = pd.DataFrame(ckpt_result)
    ckpt_result_pdf = ckpt_result_pdf[['sale_mean_reward', 'dml_mean_reward', 
                                       'single_agent_mean_reward'] + ['cv{}'.format(i) for i in range(num_kf)]]

    file_directory = os.path.join(working_directory, 'csv')
    if not os.path.isdir(file_directory):
        os.mkdir(file_directory)
    file_path = os.path.join(file_directory, 'ag_{}-sp_{}-nt_{}-kf_{}-rca_{}.csv'.format(
        agent_name, strategy, num_trajectories, num_kf, replica))
    print('Save all records to {}'.format(file_path))
    ckpt_result_pdf.to_csv(file_path, index=False, encoding='UTF-8', header=True)

    return ckpt_result_pdf

def one_round_run(replica,
    strategy = 'random',
    agent_name = 'dqn',
    n_trajs=200,
    n_splits=2,
    config_modify_func=None):

    path = 'data/{}_{}_{}/{}/trajs_qr_dqn.pkl'.format(
        agent_name, n_splits, n_trajs, strategy
    )
    kf = KFoldCV(path, n_trajs=n_trajs, n_splits=n_splits, shuffle=True, random_state=123456789, first=False)
    kf.split()
    working_directory = kf.working_directory
    print('Working directory path: {}'.format(kf.working_directory))
    print('Check point path: {}'.format(kf.ckpt_paths))


    print('Behavior Cloning...')
    bc = BehaviorCloning(num_actions=4, hiddens=[256,256], activation='relu', lr=5e-4)
    states  = np.array([transition[0] for traj in kf.trajs for transition in traj])
    actions = np.array([transition[1] for traj in kf.trajs for transition in traj])
    bc.train(states, actions)

    print('Single Agent Training...')
    if config_modify_func is not None:
        config_modify_func(kf=kf)
    else:
        set_config(kf=kf)
    print('With Config: {}'.format(config))

    agent = DQNAgent(name='LunarLander-v2', num_actions=4, config=config)
    agent.learn()

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

    file_path = 'dqn_{}_single_rc{}_dt{}.jpg'.format(strategy, replica, datetime.now().strftime('%Y%m%d_%H-%M-%s'))
    pic_dir = os.path.join(working_directory, 'pic')
    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)
    plt.savefig(os.path.join(pic_dir, file_path))

    print('Save Single Agent Training...')
    file_name = 'offline-single-lr{}-dedcay_step{}-max_training_steps{}'.format(
    config['lr'], config['decay_steps'], config['max_training_steps'])
    save_weight_dir = os.path.join(working_directory, 'model_save')
    file_path = '{file_name}.weights'.format_map({'file_name': file_name})
    agent.save(os.path.join(working_directory, file_name, file_path))

    print('Cross Validating...')
    for idx in range(kf.n_splits):
        config_idx = copy.deepcopy(config)
        config_idx['persistent_directory'] = kf.agent_paths[idx]
        config_idx['checkpoint_path'] = kf.ckpt_paths[idx]

        agent_idx = DQNAgent(name='LunarLander-v2', num_actions=4, config=config_idx)
        agent_idx.learn()

        rewards = pd.Series(agent_idx.eval_episode_rewards)
        steps = pd.Series(agent_idx.eval_episode_steps)

        fig, axes = plt.subplots(2, 2, figsize=(18, 8))

        axes[0][0].plot(rewards.rolling(100, min_periods=20).mean())
        axes[0][0].set_title('mean reward')
        axes[0][1].plot(rewards.rolling(100, min_periods=20).max())
        axes[0][1].set_title('max reward')
        axes[1][0].plot(steps.rolling(100, min_periods=20).mean())
        axes[1][0].set_title('mean step')
        axes[1][1].plot(steps.rolling(100, min_periods=20).max())
        axes[1][1].set_title('max step')

        file_path = 'dqn_{}_cv_rc{}_dt{}.jpg'.format(strategy, replica, datetime.now().strftime('%Y%m%d_%H-%M-%s'))
        plt.savefig(os.path.join(pic_dir, file_path))

    rs = compare_within_ckpt(kf, bc, config, working_directory, strategy = strategy,
                            num_trajectories = n_trajs,
                            agent_name = agent_name,
                            num_kf = n_splits, replica=replica)

if __name__ == "__main__":
    for replica in range(1, 6):
        one_round_run(replica,
            strategy = 'random',
            agent_name = 'ddqn',
            n_trajs=200,
            n_splits=2)