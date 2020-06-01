import os
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

class CVS(object):
    '''
    Cross validation, folder construction & data splitting.
    '''
    def __init__(self, path, n_splits=5, shuffle=True, random_state=123456789):
        self.path = path
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
            
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        dirname = os.path.abspath(os.path.dirname(self.path))
        
        digits = np.random.randint(1000000000, size=1)[0]
        dirname = dirname + '/tmp/' + str(digits)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        self.fold_paths = [dirname + '/fold' + str(k) for k in range(self.n_splits)]
        for path in self.fold_paths:
            if not os.path.exists(path):
                os.mkdir(path)
        
        self.train_paths = [fold_path + '/train/' for fold_path in self.fold_paths]
        for path in self.train_paths:
            if not os.path.exists(path):
                os.mkdir(path)        
        
        self.test_paths = [fold_path + '/test/' for fold_path in self.fold_paths]
        for path in self.test_paths:
            if not os.path.exists(path):
                os.mkdir(path)
        
    def split(self):
        with open(self.path, 'rb') as f:
            trajs = pickle.load(f)
        for k, (train_index, test_index) in enumerate(self.kf.split(trajs)):
            train_trajs = [trajs[index] for index in train_index]
            test_trajs = [trajs[index] for index in test_index]
            with open(self.train_paths[k] + 'trajs.pkl', 'wb') as f:
                pickle.dump(train_trajs, f)
            with open(self.test_paths[k] + 'trajs.pkl', 'wb') as f:
                pickle.dump(test_trajs, f)

class KFoldCV(object):
    def __init__(self, path, n_trajs=None, n_splits=5, shuffle=False, random_state=123456789, first=True):
        self.path = path
        self.n_trajs = n_trajs
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        with open(path, 'rb') as f:
            self.trajs = pickle.load(f)
            
        if shuffle:
            random.Random(random_state).shuffle(self.trajs)
        
        if n_trajs is not None and len(self.trajs) >= n_trajs:
            self.trajs = self.trajs[:n_trajs] if first else self.trajs[-n_trajs:]
        
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def split(self):
        dirname = os.path.abspath(os.path.dirname(self.path))
        
        agent_path = dirname + '/agent/'
        if not os.path.exists(agent_path):
            os.mkdir(agent_path)
            
        ckpt_path = agent_path + 'ckpt/'
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
            
        with open(agent_path + 'trajs.pkl', 'wb') as f:
            pickle.dump(self.trajs, f)

        agent_paths = [dirname + '/agent{}/'.format(k) for k in range(self.n_splits)]
        for path in agent_paths:
            if not os.path.exists(path):
                os.mkdir(path)
                
        ckpt_paths = [agent_path + 'ckpt/' for agent_path in agent_paths]
        for path in ckpt_paths:
            if not os.path.exists(path):
                os.mkdir(path)

        for k, (train_index, test_index) in enumerate(self.kf.split(self.trajs)):
            train_trajs = [self.trajs[index] for index in train_index]
            with open(agent_paths[k] + 'trajs{}.pkl'.format(k), 'wb') as f:
                pickle.dump(train_trajs, f)
                
        self.agent_path = agent_path           
        self.agent_paths = agent_paths
        self.ckpt_path = ckpt_path        
        self.ckpt_paths = ckpt_paths
        
    # agents, behavior cloning, density ratios
    def update_q(self, agents, bc, density_ratios=None):
        
        states_, qvalues_, qtildes_ = [], [], []
        
        for k, (train_index, test_index) in enumerate(self.kf.split(self.trajs)):
            test_trajs = [self.trajs[index] for index in test_index]
            
            states = np.array([transition[0] for traj in test_trajs for transition in traj])
            actions = np.array([transition[1] for traj in test_trajs for transition in traj])
            rewards = np.array([transition[2] for traj in test_trajs for transition in traj])
            next_states = np.array([transition[3] for traj in test_trajs for transition in traj])
            
            q_vals = agents[k].model(states).q_values.numpy()
            chosen_q_vals = q_vals[range(len(actions)), actions]
            next_vals = tf.math.reduce_max(agents[k].model(next_states).q_values, axis=1).numpy()
            td_errors = (rewards + agents[k].config['gamma'] * next_vals - chosen_q_vals).clip(-20, 20)

            pscores = bc.policy(states, actions)
            q_tildes = q_vals.copy()
            q_tildes[range(len(actions)), actions] += (td_errors / (pscores + 1e-2)).clip(-100, 100)
            
            states_.append(states)
            qvalues_.append(q_vals)
            qtildes_.append(q_tildes)
        
        states, qvalues, qtildes = np.vstack(states_), np.vstack(qvalues_), np.vstack(qtildes_)
        self.states, self.qvalues, self.qtildes = states, qvalues, qtildes

        return [states, qvalues, qtildes]