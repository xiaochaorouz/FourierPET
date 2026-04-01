"""
Optimizer and scheduler registry.
"""
from typing import Dict
import torch.optim as optim

OPTIMIZER_MAP = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
}

SCHEDULER_MAP = {
    'cosine_with_warmup': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
}


def get_optimizer(p: Dict, model):
    optimizer_name = p['optimizer']
    if optimizer_name not in OPTIMIZER_MAP:
        raise ValueError(f'Invalid optimizer: {optimizer_name}')
    return OPTIMIZER_MAP[optimizer_name](
        model.parameters(), **p['optimizer_kwargs'])


def get_scheduler(p: Dict, optimizer):
    scheduler_name = p['scheduler']
    if scheduler_name not in SCHEDULER_MAP:
        raise ValueError(f'Invalid scheduler: {scheduler_name}')
    return SCHEDULER_MAP[scheduler_name](
        optimizer, **p['scheduler_kwargs'])
