"""
Configuration loader.

Reads a YAML experiment config and sets up output directory paths.
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_exp):
    """
    Create configuration from an experiment config file.

    The config file should contain 'root_dir' (default: 'outputs') and 'exp_name'.
    Output paths for checkpoints, logs, and figures are automatically set up.
    """
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v

    root_dir = cfg.get('root_dir', 'outputs')

    base_dir = os.path.join(root_dir, cfg['exp_name'])
    mkdir_if_missing(base_dir)
    cfg['output_dir'] = base_dir
    cfg['checkpoint_last'] = os.path.join(base_dir, 'last.pth.tar')
    cfg['checkpoint'] = os.path.join(base_dir, 'checkpoint.pth.tar')
    cfg['list_log'] = os.path.join(base_dir, 'list_log.txt')
    cfg['model'] = os.path.join(base_dir, 'model.pth.tar')
    cfg['train_list_log'] = os.path.join(base_dir, 'train_list_log.txt')
    cfg['val_list_log'] = os.path.join(base_dir, 'val_list_log.txt')
    cfg['x_xhat_list_log'] = os.path.join(base_dir, 'x_xhat_list_log.txt')
    figures_dir = os.path.join(base_dir, 'figures')
    mkdir_if_missing(figures_dir)
    cfg['figures_base'] = figures_dir

    return cfg
