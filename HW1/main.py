from core.mlp import MLP
from core.config import parse_config
import sys

if sys.argv[1] != None:
    config_name = sys.argv[1]
else:
    config_name = 'config1'


config = parse_config(config_name=config_name)

mlp = MLP(config=config, name=config_name)
mlp.train()