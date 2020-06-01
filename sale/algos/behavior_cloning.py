import random
import numpy as np
import tensorflow as tf

from sale.models.basic_models import MLPNetwork

class BehaviorCloning:
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
        
        # model, optimizer, loss, callbacks
        self.model = MLPNetwork(num_actions, hiddens, activation)
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                         lr, decay_steps=decay_steps, decay_rate=1))
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        
        self.callbacks = []
        if self.validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
        
    def train(self, states, actions):
        self.history = self.model.fit(states, actions,
                                      batch_size=self.batch_size, 
                                      epochs=self.max_epoch, 
                                      verbose=self.verbose,
                                      validation_split=self.validation_split,
                                      callbacks=self.callbacks)
        
    def policy(self, states, actions):
        # states shape: (batch_size, ...), actions shape: (batch_size,)
        logits = self.model.predict(states)
        elogits = np.exp(logits)
        softmax = elogits / elogits.sum(axis=-1, keepdims=True)
        probabilities = np.squeeze(softmax[range(len(actions)), actions])
        return probabilities
        
    def greedy_actions(self, states):
        return np.argmax(self.model(states), axis=1)
    
    def _select_action(self, state, epsilon=1e-3):
        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.model(state[None]))