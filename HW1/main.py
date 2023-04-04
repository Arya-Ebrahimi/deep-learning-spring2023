from core.mlp import MLP
from core.config import *
import sys

config_name = sys.argv[1]

if config_name == 'all':
    configs, names = parse_configs()
    for i in range(len(configs)):
        mlp = MLP(config=configs[i], name=names[i])
        mlp.train()
else:
    config = parse_config(config_name=config_name)

    mlp = MLP(config=config, name=config_name)
    mlp.train()