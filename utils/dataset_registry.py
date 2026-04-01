"""
Dataset registry — manages dataset creation via the registry pattern.
"""
from typing import Dict, Callable, Any, Optional, Iterable
import torch
import torch.utils.data


class JointAugmentDataset(torch.utils.data.Dataset):
    """Wrapper that applies identical random augmentation across all views."""

    def __init__(self, dataset, augment_transform=None, exclude_keys=None):
        self.dataset = dataset
        self.augment_transform = augment_transform
        self.exclude_keys = set(exclude_keys or [])

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.augment_transform is None:
            return sample

        if isinstance(sample, dict):
            sample = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            candidate_keys = [
                key for key, value in sample.items()
                if key not in self.exclude_keys
                and torch.is_tensor(value)
                and value is not None
                and value.ndim >= 3
            ]
            if not candidate_keys:
                return sample

            stacked = torch.stack(
                [sample[key] for key in candidate_keys], dim=0)
            augmented = self.augment_transform(stacked)
            if augmented.ndim == 3:
                augmented = augmented.unsqueeze(0)
            for idx, key in enumerate(candidate_keys):
                sample[key] = augmented[idx]
            return sample

        if torch.is_tensor(sample) and sample.ndim >= 3:
            stacked = sample.unsqueeze(0)
            augmented = self.augment_transform(stacked)
            if augmented.ndim == 4:
                augmented = augmented.squeeze(0)
            return augmented

        return sample


def get_pet_joint_augment(image_size=32):
    from torchvision.transforms.v2 import (
        Compose, RandomAffine, RandomApply,
        RandomHorizontalFlip, RandomRotation, ToDtype, GaussianBlur,
    )
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomApply([RandomRotation(degrees=10)], p=0.3),
        RandomApply(
            [RandomAffine(degrees=0, translate=(0.05, 0.05),
                          scale=(0.9, 1.1))],
            p=0.3,
        ),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        ToDtype(torch.float32, scale=False),
    ])


class DatasetRegistry:
    def __init__(self):
        self._train_registry: Dict[str, Callable] = {}
        self._val_registry: Dict[str, Callable] = {}

    def register_train(self, name: str):
        def decorator(func: Callable):
            self._train_registry[name] = func
            return func
        return decorator

    def register_val(self, name: str):
        def decorator(func: Callable):
            self._val_registry[name] = func
            return func
        return decorator

    def create_train_dataset(self, name, p, pbeam, transform):
        if name not in self._train_registry:
            raise ValueError(f'Unknown train dataset: {name}')
        dataset = self._train_registry[name](p, pbeam, transform)
        if p.get('do_augment', False):
            augment_transform = get_pet_joint_augment(
                image_size=p.get('image_size', 128))
            dataset = JointAugmentDataset(
                dataset, augment_transform=augment_transform,
                exclude_keys={'prior'})
        return dataset

    def create_val_dataset(self, name, p, pbeam, transform=None):
        if name not in self._val_registry:
            raise ValueError(f'Unknown val dataset: {name}')
        return self._val_registry[name](p, pbeam, transform)


dataset_registry = DatasetRegistry()


# ==================== Real Clinical Data ====================

@dataset_registry.register_train('Real_clinical_data')
def create_real_clinical_train(p, pbeam, transform):
    from data.pet_data import RealClinicalDataset
    return RealClinicalDataset(
        dose_paths=p['dose_paths'],
        prior_path=p.get('prior_path'),
        projection=pbeam,
        original_resolution=p.get('original_resolution', 256),
        image_size=p.get('image_size', 128),
        do_augment=p.get('do_augment', False),
        do_normalize=p.get('do_normalize', False),
        Anscobe_normalize=False,
        minmax_normalize=False,
        SUV_window_threshold=p.get('threshold', 1.0),
        lmdb_zfill=p.get('lmdb_zfill', 6),
    )


@dataset_registry.register_val('Real_clinical_data')
def create_real_clinical_val(p, pbeam, transform=None):
    from data.pet_data import RealClinicalDataset
    val_dose_paths = p.get('val_dose_paths', p['dose_paths'])
    return RealClinicalDataset(
        dose_paths=val_dose_paths,
        prior_path=p.get('val_prior_path', p.get('prior_path')),
        projection=pbeam,
        original_resolution=p.get('original_resolution', 256),
        image_size=p.get('val_image_size', p.get('image_size', 128)),
        do_augment=False,
        do_normalize=p.get('do_normalize', False),
        Anscobe_normalize=False,
        minmax_normalize=False,
        SUV_window_threshold=p.get('threshold', 3.0),
        lmdb_zfill=p.get('lmdb_zfill', 6),
    )


# ==================== Simulated Data ====================

@dataset_registry.register_train('Simulated_data')
def create_simulated_train(p, pbeam, transform):
    from data.pet_data import SimulatedDataset
    load_keys = p.get('load_keys', ['full'])
    return SimulatedDataset(
        pet_path=p.get('pet_path', 'datasets/brainweb_pet.lmdb'),
        prior_path=p.get('prior_path', 'datasets/brainweb_mr.lmdb'),
        projection=pbeam,
        original_resolution=p.get('original_resolution', 256),
        image_size=p.get('image_size', 128),
        do_augment=False,
        do_normalize=p.get('do_normalize', True),
        W=p.get('threshold', 10.0),
        lmdb_zfill=p.get('lmdb_zfill', 6),
        load_keys=load_keys,
    )


@dataset_registry.register_val('Simulated_data')
def create_simulated_val(p, pbeam, transform=None):
    from data.pet_data import SimulatedDataset
    load_keys = p.get('load_keys', ['full'])
    return SimulatedDataset(
        pet_path=p.get('pet_path', 'datasets/brainweb_pet.lmdb'),
        prior_path=p.get('prior_path', 'datasets/brainweb_mr.lmdb'),
        projection=pbeam,
        original_resolution=p.get('original_resolution', 256),
        image_size=p.get('val_image_size', p.get('image_size', 128)),
        do_augment=False,
        do_normalize=p.get('do_normalize', True),
        W=p.get('threshold', 10.0),
        lmdb_zfill=p.get('lmdb_zfill', 6),
        load_keys=load_keys,
    )
