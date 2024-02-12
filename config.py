from easydict import EasyDict as edict
import yaml

def get_config(config_path, overrides=[]):
    with open(config_path, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = edict(yaml_cfg)

    for override in overrides:
        key, value = override.split("=")
        if '.' in key:
            keys = key.split('.')
            subconfig = config
            for k in keys[:-1]:
                subconfig = subconfig[k]
            subconfig[keys[-1]] = type(subconfig[keys[-1]])(value)
        else:
            config[key] = type(config[key])(value)
    return config
