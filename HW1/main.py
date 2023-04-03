from core.mlp import MLP
from core.config import parse_config

config_name = 'config1'


config = parse_config(config_name=config_name)

mlp = MLP(config=config)
mlp.train()