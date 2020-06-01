import collections
import numpy as np
import tensorflow as tf

DQNNetworkType = collections.namedtuple(
    'dqn_network', ['q_values'])
QuantileNetworkType = collections.namedtuple(
    'qr_dqn_network', ['q_values', 'logits', 'probabilities'])
MultiHeadNetworkType = collections.namedtuple(
    'multi_head_dqn_network', ['q_heads', 'unordered_q_heads', 'q_values'])
MultiNetworkNetworkType = collections.namedtuple(
    'multi_network_dqn_network', ['q_networks', 'unordered_q_networks', 'q_values'])

class DQNNetwork(tf.keras.Model):
    def __init__(self, num_actions, 
                 hiddens=[64,64], activation='relu', 
                 dueling=False, name='dqn_network'):
        super().__init__(name=name)
        
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        self.dueling = dueling
        # defining layers
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        output_units = num_actions + 1 if dueling else num_actions
        dense_layer = tf.keras.layers.Dense(units=output_units, activation=None)
        self.dense_layers.append(dense_layer)
        
    def call(self, state):
        net = tf.cast(state, tf.float32)
        for dense in self.dense_layers:
            net = dense(net)
        if self.dueling:
            net = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:,0],-1) + 
                                         x[:,1:] - tf.math.reduce_mean(x[:,1:], keepdims=True))(net)
        return DQNNetworkType(net)
    
class QuantileNetwork(tf.keras.Model):
    def __init__(self, num_actions, num_atoms=51, 
                 hiddens=[64,64], activation='relu', 
                 name='qr_dqn_network'):
        super(QuantileNetwork, self).__init__(name=name)
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.hiddens = hiddens
        self.activation = activation
        # defining layers
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        dense_layer = tf.keras.layers.Dense(units=num_actions * num_atoms, activation=None)
        self.dense_layers.append(dense_layer)

    def call(self, state):
        """Calculates the distribution of Q-values using the input state tensor."""
        net = tf.cast(state, tf.float32)
        for dense in self.dense_layers:
            net = dense(net)
        logits = tf.reshape(net, [-1, self.num_actions, self.num_atoms])
        probabilities = tf.keras.activations.softmax(tf.zeros_like(logits))
        q_values = tf.reduce_mean(logits, axis=2)
        return QuantileNetworkType(q_values, logits, probabilities)

def random_stochastic_matrix(num_heads, num_convex_combinations=None, dtype=tf.float32):
    """Generates a random left stochastic matrix."""
    mat_shape = (num_heads, num_heads) if num_convex_combinations is None else (num_heads, num_convex_combinations)
    mat = tf.random.uniform(shape=mat_shape, dtype=dtype)
    mat /= tf.norm(mat, ord=1, axis=0, keepdims=True)
    return mat

def combine_q_functions(q_functions, transform_strategy, transform_matrix=None):
    """
    Utility function for combining multiple Q functions.
    Args:
    q_functions: Multiple Q-functions concatenated.
    transform_strategy: str, Possible options include (1) 'IDENTITY' for no
      transformation (2) 'STOCHASTIC' for random convex combination.
    Returns:
    q_functions: Modified Q-functions.
    q_values: Q-values based on combining the multiple heads.
    """
    # Create q_values before reordering the heads for training
    q_values = tf.reduce_mean(q_functions, axis=-1)

    if transform_strategy == 'STOCHASTIC':
        if transform_matrix is None:
            raise ValueError('None value provided for transform matrix')
        q_functions = tf.tensordot(q_functions, transform_matrix, axes=[[2], [0]])
    elif transform_strategy == 'IDENTITY':
        tf.logging.info('Identity transformation Q-function heads')
    else:
        raise ValueError('{} is not a valid reordering strategy'.format(transform_strategy))
    return q_functions, q_values

class MultiHeadQNetwork(tf.keras.Model):
    def __init__(self, num_actions, hiddens=[64,64], activation='relu',
                 num_heads=5, num_convex_combinations=5,                 
                 transform_strategy='STOCHASTIC',
                 name='multi_head_dqn_network'):
        """
        Creates the layers used calculating return distributions.
        Args:
          num_actions: number of actions.
          num_heads: number of Q-heads.
          transform_strategy: Possible options include (1) 'IDENTITY' for no
            transformation (Ensemble-DQN) (2) 'STOCHASTIC' for random convex
            combination (REM).
          name: used to create scope for network parameters.
        """
        super(MultiHeadQNetwork, self).__init__(name=name)
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        self.num_heads = num_heads
        
        self._transform_strategy = transform_strategy
        self._num_convex_combinations = num_convex_combinations if num_convex_combinations else num_heads
        self._transform_matrix = random_stochastic_matrix(self.num_heads,
                                                          self._num_convex_combinations)
        # defining layers
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        dense_layer = tf.keras.layers.Dense(units=num_actions * num_heads, activation=None)
        self.dense_layers.append(dense_layer)

    def call(self, state):
        net = tf.cast(state, tf.float32)
        for dense in self.dense_layers:
            net = dense(net)
        unordered_q_heads = tf.reshape(net, [-1, self.num_actions, self.num_heads])
        q_heads, q_values = combine_q_functions(
            unordered_q_heads, self._transform_strategy, self._transform_matrix)
        return MultiHeadNetworkType(q_heads, unordered_q_heads, q_values)

class MultiNetworkQNetwork(tf.keras.Model):
    def __init__(self, num_actions, hiddens=[64,64], 
                 activation='relu', dueling=False, 
                 num_networks=5, num_convex_combinations=5,
                 transform_strategy='STOCHASTIC',
                 name='multi_network_dqn_network'):
        """
        Creates the networks used calculating multiple Q-values.
        Args:
          num_actions: number of actions.
          num_networks: number of separate Q-networks.
          transform_strategy: Possible options include (1) 'IDENTITY' for no
            transformation (Ensemble-DQN) (2) 'STOCHASTIC' for random convex
            combination (REM).
          name: used to create scope for network parameters.
        """
        super(MultiNetworkQNetwork, self).__init__(name=name)
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        self.dueling = dueling
        
        self.num_networks = num_networks
        self._transform_strategy = transform_strategy
        self._num_convex_combinations = num_convex_combinations if num_convex_combinations else num_networks
        self._transform_matrix = random_stochastic_matrix(self.num_networks, self._num_convex_combinations)
        # Create multiple Q-networks
        self._q_networks = []
        for i in range(self.num_networks):
            q_net = DQNNetwork(num_actions, hiddens,
                               activation, dueling,
                               name='subnet_{}'.format(i))
            self._q_networks.append(q_net)

    def call(self, state):
        """
        Creates the output tensor/op given the input state tensor.
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.
        Args:
          state: Tensor, input tensor.
        Returns:
          collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        unordered_q_networks = [network(state).q_values for network in self._q_networks]
        unordered_q_networks = tf.stack(unordered_q_networks, axis=-1)
        q_networks, q_values = combine_q_functions(unordered_q_networks,
                                                   self._transform_strategy,
                                                   self._transform_matrix)
        return MultiNetworkNetworkType(q_networks, unordered_q_networks, q_values)