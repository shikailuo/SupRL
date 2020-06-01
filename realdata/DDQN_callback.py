import os
import copy
import random
import gym
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

from itertools import permutations
from sklearn.model_selection import KFold, GridSearchCV

from multiprocessing import set_start_method
import multiprocessing as mp

path = os.path.abspath('..')
if path not in sys.path:
    sys.path.append(path)

from sale.agents.default_config import DEFAULT_CONFIG as config
from sale.agents.dqn import DQNAgent
from sale.agents.qr_dqn import QuantileAgent

from sale.algos.kfold import CVS, KFoldCV
from sale.algos.advantage_learner import AdvantageLearner
from sale.algos.behavior_cloning import BehaviorCloning
from sale.algos.density_ratio import VisitationRatioModel
from sale.algos.fqe import FQE

def one_step(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    path = 'data/callback/trajs_city_id_5_reward_type_4.pkl'
    nfolds = 5
    n_splits = 5
    ckpts = (np.arange(10) + 1)*5000
    
    num_actions = 6
    # configures
    config['online'] = False
    config['hiddens'] = [64,64]
    config['double'] = True
    config['dueling'] = False
    config['lr'] = 5e-4
    config['decay_steps'] = 50000
    config['max_training_steps'] = 50000
    config['training_steps_to_checkpoint'] = 5000
    config['training_steps_to_eval'] = 100000

    index = pd.MultiIndex.from_product([np.arange(nfolds), ckpts])
    columns = ['dqn', 'dml', 'sale']
    rets = pd.DataFrame(index=index, columns=columns)

    print('-'*20, 'start', '-'*20)

    cvs = CVS(path, n_splits=nfolds, random_state=seed)
    cvs.split()
    for fold in range(nfolds):
        train_path = cvs.train_paths[fold] + 'trajs.pkl'
        kf = KFoldCV(train_path, n_trajs=None, n_splits=n_splits, shuffle=False, random_state=seed)
        kf.split()

        print('-'*20, 'training agent', '-'*20)
        # agent
        config['persistent_directory'] = kf.agent_path
        config['checkpoint_path'] = kf.ckpt_path
        agent = DQNAgent(num_actions=num_actions, config=config)
        agent.learn()

        print('-'*20, 'training agents', '-'*20)
        # agent_1, ..., agent_K
        for idx in range(kf.n_splits):
            config_idx = copy.deepcopy(config)
            config_idx['persistent_directory'] = kf.agent_paths[idx]
            config_idx['checkpoint_path'] = kf.ckpt_paths[idx]
            agent_idx = DQNAgent(num_actions=num_actions, config=config_idx)
            agent_idx.learn()

        # fitted q evaluation
        test_path = cvs.test_paths[fold] + 'trajs.pkl'
        with open(test_path, 'rb') as f:
            trajs = pickle.load(f)

        print('-'*20, 'behavior cloning', '-'*20)
        # behavior cloning
        bc = BehaviorCloning(num_actions=num_actions)
        states  = np.array([transition[0] for traj in kf.trajs for transition in traj])
        actions = np.array([transition[1] for traj in kf.trajs for transition in traj])
        bc.train(states, actions)

        for ckpt in ckpts:
            print('-'*20, 'ckpt: ', ckpt, '-'*20)
            agent = DQNAgent(num_actions=num_actions, config=config)
            agent.load(kf.ckpt_path + 'dqn_{}.ckpt'.format(ckpt))

            agents = []
            for idx in range(kf.n_splits):
                config_idx = copy.deepcopy(config)
                config_idx['persistent_directory'] = kf.agent_paths[idx]
                config_idx['checkpoint_path'] = kf.ckpt_paths[idx]
                agent_idx = DQNAgent(num_actions=num_actions, config=config_idx)
                agent_idx.load(kf.ckpt_paths[idx] + 'dqn_{}.ckpt'.format(ckpt))
                agents.append(agent_idx)
            states, qvalues, qtildes = kf.update_q(agents, bc)

            print('-'*20, 'adv learner', '-'*20)
            advs1 = qvalues - qvalues.mean(axis=1, keepdims=True)
            agent1 = AdvantageLearner(num_actions=num_actions)
            agent1._train(states, advs1)
            
            advs2 = qtildes - qtildes.mean(axis=1, keepdims=True)
            agent2 = AdvantageLearner(num_actions=num_actions)
            agent2._train(states, advs2)

            print('-'*20, 'fqe on dqn & dml & sale', '-'*20)
            fqe_dqn = FQE(agent.greedy_actions, num_actions=num_actions)
            fqe_dqn.train(trajs)
            fqe_dml = FQE(agent1.greedy_actions, num_actions=num_actions)
            fqe_dml.train(trajs)
            fqe_sale = FQE(agent2.greedy_actions, num_actions=num_actions)
            fqe_sale.train(trajs)

            rets.loc[(fold, ckpt), 'dqn'] = fqe_dqn.values
            rets.loc[(fold, ckpt), 'dml'] = fqe_dml.values
            rets.loc[(fold, ckpt), 'sale'] = fqe_sale.values
            
    return rets

save_path = 'rets/'
pool = mp.Pool(10)
rets = pool.map(one_step, range(10))
pool.close()

with open(save_path + 'rets_ddqn_callback_5_4.pkl', 'wb') as f:
    pickle.dump(rets, f)