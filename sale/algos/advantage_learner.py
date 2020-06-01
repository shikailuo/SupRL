import random
import gym
import numpy as np
import tensorflow as tf

from sale.models.basic_models import MLPNetwork

class AdvantageLearner:
    def __init__(self, num_actions=4,
                 hiddens=[256,256], activation='relu',
                 lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200,
                 validation_split=0.2, patience=20,
                 verbose=0):
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
        if self.validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]        

    def _train(self, states, advs):
        self.history = self.model.fit(states, advs,
                                      batch_size=self.batch_size, 
                                      epochs=self.max_epoch,
                                      verbose=self.verbose,
                                      validation_split=self.validation_split,
                                      callbacks=self.callbacks)

    def _select_action(self, state, epsilon=1e-3): # for state
        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.model(state[None]), axis=1)[0]

    def greedy_actions(self, states):
        return np.argmax(self.model(states), axis=1)
        
    def _eval(self, n_episodes, env=gym.make('LunarLander-v2'), max_episode_steps=1000):
        self.eval_episode_rewards = []
        self.eval_episode_steps = []
        for i in range(n_episodes):
            rewards, steps = 0, 0
            state = env.reset()
            for t in range(max_episode_steps):
                action = self._select_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                rewards += reward
                steps += 1
                if done: break       
            self.eval_episode_rewards.append(rewards)
            self.eval_episode_steps.append(steps)