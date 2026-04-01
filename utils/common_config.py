"""
Common configuration module — factory functions for creating training objects.
"""
import torch
import torchvision.transforms as transforms
import numpy as np

try:
    from .dataset_registry import dataset_registry
    from .model_registry import model_registry
    from .criterion_registry import get_criterion
    from .optimizer_registry import get_optimizer, get_scheduler
except ImportError:
    from utils.dataset_registry import dataset_registry
    from utils.model_registry import model_registry
    from utils.criterion_registry import get_criterion
    from utils.optimizer_registry import get_optimizer, get_scheduler


def get_train_dataset(p, pbeam, transform):
    return dataset_registry.create_train_dataset(
        p['train_db_name'], p, pbeam, transform)


def get_val_dataset(p, pbeam, transform=None):
    return dataset_registry.create_val_dataset(
        p['val_db_name'], p, pbeam, transform)


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(
        dataset, num_workers=p['num_workers'], batch_size=p['batch_size'],
        pin_memory=True, drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(
        dataset, num_workers=p['num_workers'], batch_size=p['batch_size'],
        pin_memory=True, drop_last=False, shuffle=False)


def get_train_dataloader_LOOCV(p, dataset, leave_out_case,
                               slices_per_case=81):
    total_slices = len(dataset)
    num_cases = total_slices // slices_per_case
    assert 0 <= leave_out_case < num_cases, \
        f"leave_out_case={leave_out_case} out of range: dataset has {total_slices} slices, " \
        f"slices_per_case={slices_per_case} -> {num_cases} cases. " \
        f"Check 'slices_per_case' and 'LOOCV_num' in your config."

    start_idx = leave_out_case * slices_per_case
    end_idx = start_idx + slices_per_case
    train_indices = list(range(0, start_idx)) + list(range(end_idx, total_slices))
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    return torch.utils.data.DataLoader(
        train_subset, batch_size=p['batch_size'], shuffle=True,
        num_workers=p['num_workers'], pin_memory=True, drop_last=True)


def get_val_dataloader_LOOCV(p, dataset, leave_out_case,
                             slices_per_case=81):
    total_slices = len(dataset)
    num_cases = total_slices // slices_per_case
    assert 0 <= leave_out_case < num_cases, \
        f"leave_out_case={leave_out_case} out of range: dataset has {total_slices} slices, " \
        f"slices_per_case={slices_per_case} -> {num_cases} cases. " \
        f"Check 'slices_per_case' and 'LOOCV_num' in your config."

    start_idx = leave_out_case * slices_per_case
    end_idx = start_idx + slices_per_case
    val_indices = list(range(start_idx, end_idx))
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    return torch.utils.data.DataLoader(
        val_subset, batch_size=p['batch_size'], shuffle=False,
        num_workers=p['num_workers'], pin_memory=True, drop_last=False)


def get_projection_simulation(p):
    if p['projection_type'] != 'parallel_beam':
        raise ValueError(f'Invalid projection type: {p["projection_type"]}')
    from torch_radon import ParallelBeam
    projection_config = p['projection_kwargs']
    angles = np.linspace(
        projection_config['angles'][0] * np.pi,
        projection_config['angles'][1] * np.pi,
        projection_config['angles'][2],
        endpoint=False)
    return ParallelBeam(
        det_count=projection_config['del_count'],
        angles=angles,
        volume=projection_config['volume'])


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'classical_noise':
        return transforms.Compose([transforms.ToTensor()])
    return None


def get_val_transformations(p):
    return None


def get_model(p):
    return model_registry.create_model(p['backbone'], p)


__all__ = [
    'get_train_dataset', 'get_val_dataset',
    'get_train_dataloader', 'get_val_dataloader',
    'get_train_dataloader_LOOCV', 'get_val_dataloader_LOOCV',
    'get_projection_simulation',
    'get_train_transformations', 'get_val_transformations',
    'get_model', 'get_criterion', 'get_optimizer', 'get_scheduler',
]
