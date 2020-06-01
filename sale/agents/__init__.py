from sale.agents.dqn import DQNAgent
from sale.agents.qr_dqn import QuantileAgent
from sale.agents.multi_head_dqn import MultiHeadDQNAgent
from sale.agents.default_config import DEFAULT_CONFIG

__all__ = ['DQNAgent',
           'QuantileAgent', 
           'MultiHeadDQNAgent', 
           'DEFAULT_CONFIG']