from sale.utils.epsilon_decay import linearly_decaying_epsilon

DEFAULT_CONFIG = {
    # === quantile dqn network ===
    "num_atoms": 51,
    # === multi head network ===
    "num_heads": 10,
    "num_convex_combinations": 10, 
    # === common config ===
    "online": True,
    "gamma": 0.99,
    "max_episode_steps": 1000,

    # === Model ===
    # whether to use dueling dqn
    "dueling": True,
    # default hiddens structure
    "hiddens": [256,256],
    # whether to use double dqn
    "double": True,
    # activation function
    "activation": 'relu',

    # === Exploration Settings ===
    "epsilon_fn": linearly_decaying_epsilon,
    # start training epsilon
    "epsilon_start": 0.1,
    "epsilon_decay_period": 100000,
    # end training epsilon
    "epsilon_end": 0.1,
    # whether in evaluation mode
    "eval_mode": False,
    # epsilon-greedy in evaluation mode
    "epsilon_eval": 0.001,

    # === Replay buffer ===    
    "buffer_size": 100000,
    # learning starts
    "min_replay_history": 1000, 
    # if True prioritized replay buffer will be used.
    "prioritized_replay": False,
    # alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # directory from which to save and load trajs
    "persistent_directory": '/tmp/',
    # save to disk every `episode_counts_to_save` trajs when online
    "episode_counts_to_save": 100,
    # load to buffer every `sample steps to refresh` training steps when offline
    "sample_steps_to_refresh": 1000000,

    # === Optimization ===
    # training batch size
    "batch_size": 64,
    # learning rate for adam optimizer
    "lr": 5e-4,
    # learning rate schedule - decay steps
    "decay_steps": 1000000,
    # if not None, clip gradients during optimization at this value
    "grad_clip": 40,
    # reward clipping
    "reward_clip": 200,
    "max_training_steps": 1000000,
    "training_steps_to_eval": 1000,
    
    # soft-update target model params
    "tau": 0.999,
    # training every `update_period` steps
    "update_period": 1,
    # hard-update the target network every `target_network_update_freq` steps
    "target_update_period": 1,
    
    # stopping critera
    "target_mean_episode_reward": 500,
    # checkpoint frequency
    "training_steps_to_checkpoint": 10000,
    # checkpoint path
    "checkpoint_path": '/tmp/'
}