import numpy as np

def linearly_decaying_epsilon(current_step, epsilon_start, decay_period, epsilon_end, warmup_steps=0):
    '''
    exploration schedules
        linearly decay from epsilon_start to epsilon_end in 'decay_period' steps,
        no decay before 'warmup_steps' steps.
    current_step: int
        current training step
    epsilon_start: float
        initial exploration epsilon
    decay_period: int
        ususally equal to maximum training steps
    epsilon_end: float
        final exploration epsilon
    warmup_steps: int
        no decay before this number of steps
    '''
    steps_left = decay_period + warmup_steps - current_step
    bonus = (epsilon_start - epsilon_end) * steps_left / decay_period
    bonus = np.clip(bonus, 0., epsilon_start - epsilon_end)
    return epsilon_end + bonus