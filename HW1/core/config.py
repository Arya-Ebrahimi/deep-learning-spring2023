import json
import glob
import re

def parse_config(config_name):
    file = open('configs/'+config_name+'.json')
    config = json.load(file)
    return config

def parse_configs():
    configs = glob.glob('configs/*.json')
    ret_list = []
    names = []
    for conf in configs:
        file = open(conf)
        config = json.load(file)
        ret_list.append(config)
        a = re.findall(r"\bc\w*\d", conf)[0]
        names.append(a)
                
    return ret_list, names
