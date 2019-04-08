"""
Configuration functions
"""
from easydict import EasyDict
import yaml

from config.defaults import get_default


def update_config(config_file, default_config):
    with open(config_file) as f:
        file_config = EasyDict(yaml.load(f))
        recursive_update(file_config, default_config)


def recursive_update(file_config, default_config):
    for ki, vi in file_config.items():
        if isinstance(vi, EasyDict):
            recursive_update(vi, default_config[ki])
        else:
            default_config[ki] = vi


def check_config(in_config, k=''):
    # recursive function to check for no Nones...
    # All default Nones need to be specified in experimental .yaml files
    for ki, vi in in_config.items():
        if isinstance(vi, EasyDict):
            check_config(vi, k+'.'+ki)
        elif vi is None:
            raise ValueError("%s must be specified in the .yaml config file" % (k+'.'+ki))
        elif vi == '':
            in_config[ki] = None


def load_config(config_file):
    default_config = get_default()
    update_config(config_file, default_config)
    return default_config
