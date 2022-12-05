import  os
import json

def get_config(config_path):

    with open(os.path.join(config_path, 'config.json'),encoding='utf8') as f1:
        config = json.load(f1)


    return config