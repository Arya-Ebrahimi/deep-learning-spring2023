import json

def parse_config(config_name):
    file = open('configs/'+config_name+'.json')
    config = json.load(file)
    return config