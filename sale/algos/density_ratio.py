import numpy as np
import tensorflow as tf

class VisitationRatioModel:
    def __init__(self, model, optimizer, replay_buffer,
                 target_policy, behavior_policy, medians=None):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.medians = medians
        self.losses = []

    def _compute_medians(self, n=100):
        transitions = self.replay_buffer.sample(n)
        states, actions, next_states = transitions[0], transitions[1], transitions[3]
        mat = tf.concat([states, actions[:,np.newaxis], next_states], axis=-1)  # n x ...
        dxx = tf.repeat(mat, n, axis=0) - tf.tile(mat, [n, 1])                  # n2 x ...
        medians = np.median(tf.math.abs(dxx), axis=0) + 1e-2 # p
        return medians
    
    def _normalize(self, weights, batch_size):
        weights = tf.reshape(weights, [batch_size, batch_size])
        weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
        weights = weights / weights_sum
        return tf.reshape(weights, [batch_size**2])
        
    def _compute_loss(self, states, actions, next_states, gamma):
        batch_size = states.shape[0]
        
        states_r = tf.repeat(states, batch_size, axis=0)      # n2 x ...
        actions_r = tf.repeat(actions, batch_size)            # n2 
        states_t = tf.tile(states, [batch_size, 1])           # n2 x ...
        next_states_t = tf.tile(next_states, [batch_size, 1]) # n2 x ...
        
        ### state visitation ratios & policy ratios & deltas
        weights = self.model(states_r, actions_r, states_t)           # n2
        next_weights = self.model(states_r, actions_r, next_states_t) # n2
        weights = self._normalize(weights, batch_size)                # n2
        next_weights = self._normalize(next_weights, batch_size)      # n2 
        policy_ratios = self.target_policy(states, actions) / (self.behavior_policy(states, actions)+1e-3) # n
        policy_ratios = tf.tile(policy_ratios, [batch_size])          # n2
        policy_ratios = tf.cast(policy_ratios, weights.dtype)
        deltas = gamma * weights * policy_ratios - next_weights       # n2
        
        ### kernels
        actions_r = tf.cast(actions_r, states.dtype)
        mat1 = tf.concat([states, actions[:,None], next_states], axis=-1)        # n x ...
        mat2 = tf.concat([states_r, actions_r[:,None], next_states_t], axis=-1)  # n2 x ...
        dxx1 = tf.repeat(mat2, batch_size**2, axis=0) - tf.tile(mat2, [batch_size**2, 1]) # n4 x ...
        dxx2 = tf.repeat(mat1, batch_size**2, axis=0) - tf.tile(mat2, [batch_size, 1])    # n3 x ...
        dxx3 = tf.repeat(mat1, batch_size, axis=0)    - tf.tile(mat1, [batch_size, 1])    # n2 x ...
        dxx1 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx1)/self.medians, axis=-1)) # n4
        dxx2 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx2)/self.medians, axis=-1)) # n3
        dxx3 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx3)/self.medians, axis=-1)) # n2
        
        ### final loss
        dxx1 = tf.repeat(deltas, batch_size**2) * tf.tile(deltas, [batch_size**2]) * dxx1
        dxx2 = tf.tile(deltas, [batch_size]) * dxx2
        loss = tf.reduce_sum(dxx1)/batch_size**4 + \
               2*(1-gamma)*tf.reduce_sum(dxx2)/batch_size**3 + \
               (1-gamma)**2*tf.reduce_sum(dxx3)/batch_size**2
        
        return loss
        
    def fit(self, batch_size=32, gamma=0.99, max_iter=100):
        if self.medians is None:
            self.medians = self._compute_medians()
            
        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            states, actions, next_states = transitions[0], transitions[1], transitions[3]
            ##### compute loss function #####
            with tf.GradientTape() as tape:
                loss = self._compute_loss(states, actions, next_states, gamma)
            dw = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
            
            self.losses.append(loss.numpy())