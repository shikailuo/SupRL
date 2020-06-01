import numpy as np
import tensorflow as tf
from sale.models.basic_models import MLPNetwork

def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)

class FQE(object):
    def __init__(self, policy, # policy to be evaluated
                 num_actions=4, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99,
                 lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, 
                 validation_split=0.2, patience=20, 
                 verbose=0, max_iter=100, eps=0.001):
        self.policy = policy
        ### === network ===
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        
        ### model, optimizer, loss, callbacks ###
        self.model = MLPNetwork(num_actions, hiddens, activation)
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                         lr, decay_steps=decay_steps, decay_rate=1))
        self.model.compile(loss='huber_loss', optimizer=self.optimizer, metrics=['mae'])
        
        self.callbacks = []
        if validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]        

        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        
    def train(self, trajs, idx=0):
        states = np.array([item[0] for traj in trajs for item in traj])
        actions = np.array([item[1] for traj in trajs for item in traj])
        rewards = np.array([item[2] for traj in trajs for item in traj])
        next_states = np.array([item[3] for traj in trajs for item in traj])
        dones = np.array([item[4] for traj in trajs for item in traj])
        
        states0 = np.array([traj[idx][0] for traj in trajs])
        
        # fitted Q evaluations
        old_targets = rewards/(1-self.gamma)
        self.model.fit(states, old_targets, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epoch, 
                       verbose=self.verbose,
                       validation_split=self.validation_split,
                       callbacks=self.callbacks)
        for iteration in range(self.max_iter):
            _actions = self.policy(next_states)
            q_next_states = self.model.predict(next_states)
            targets = rewards + self.gamma * q_next_states[range(len(_actions)), _actions]*(1.0-dones)
            ## model targets
            _targets = self.model.predict(states)
            _targets[range(len(actions)), actions] = targets
            
            self.model.fit(states, _targets, 
                           batch_size=self.batch_size, 
                           epochs=self.max_epoch, 
                           verbose=self.verbose,
                           validation_split=self.validation_split,
                           callbacks=self.callbacks)
            
            self.values.append(self.state_value(states0))
            
            target_diff = change_rate(old_targets, targets)
            self.target_diffs.append(target_diff)
            print('-----iteration: ', iteration, 
                  'target diff: ', target_diff, 
                  'values: ', self.values[-1].mean(), '-----', '\n')
            
            if target_diff < self.eps:
                break
                
            old_targets = targets.copy()
            
    def state_value(self, states):
        return np.amax(self.model.predict(states), axis=1)
    
    def state_action_value(self, states):
        return self.model.predict(states)
    
    def compute_value(self, trajs, idx=0):
        states = np.array([traj[idx][0] for traj in trajs])
        return self.state_value(states)