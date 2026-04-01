"""
PET data loading module.

Provides LMDB-based dataset classes for real clinical and simulated PET data.
"""
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import lmdb
import pickle
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import threading

_lmdb_envs = {}
_lmdb_envs_lock = threading.Lock()


class BaseLMDB(Dataset):
    """Base LMDB dataset loader with multi-process support."""

    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.lmdb_path = os.path.abspath(path)
        self.length = None

    def _get_env(self):
        process_id = os.getpid()
        env_key = (process_id, self.lmdb_path)
        with _lmdb_envs_lock:
            if env_key not in _lmdb_envs:
                env = lmdb.open(
                    self.lmdb_path, max_readers=32, readonly=True,
                    lock=False, readahead=False, meminit=False)
                if not env:
                    raise IOError('Cannot open lmdb dataset', self.lmdb_path)
                _lmdb_envs[env_key] = env
                if self.length is None:
                    with env.begin(write=False) as txn:
                        length_str = txn.get('length'.encode('utf-8'))
                        if length_str:
                            self.length = int(length_str.decode('utf-8'))
                        else:
                            self.length = env.stat()['entries']
            return _lmdb_envs[env_key]

    def __getstate__(self):
        return {
            'original_resolution': self.original_resolution,
            'zfill': self.zfill,
            'lmdb_path': self.lmdb_path,
            'length': self.length,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        if self.length is None:
            try:
                with lmdb.open(self.lmdb_path, readonly=True,
                               lock=False) as env:
                    with env.begin(write=False) as txn:
                        length_str = txn.get('length'.encode('utf-8'))
                        if length_str:
                            self.length = int(length_str.decode('utf-8'))
                        else:
                            self.length = env.stat()['entries']
            except Exception:
                self.length = 0
        return self.length if self.length is not None else 0

    def __getitem__(self, index):
        env = self._get_env()
        with env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode('utf-8')
            serialized_content = txn.get(key)
            if serialized_content is None:
                return None
            content = pickle.loads(serialized_content)
            return Image.fromarray(content)


class RealClinicalDataset(Dataset):
    """Real clinical PET dataset with multiple dose levels.

    Each dose level is stored as a separate LMDB. The returned dict has the
    same keys as ``dose_paths`` plus an optional ``'prior'`` key.

    Example config::

        dose_paths:
            full: /path/to/full_dose.lmdb        # required
            5min: /path/to/5min.lmdb
            1min: /path/to/1min.lmdb
            6s:   /path/to/6s.lmdb
        prior_path: /path/to/CT.lmdb              # optional

    Args:
        dose_paths: Dict mapping dose-level names to LMDB paths.
                    Must contain a ``'full'`` key.
        prior_path: Path to structural prior LMDB (e.g. CT / MR).
        image_size: Target image size after resizing.
        original_resolution: Resolution stored in LMDB keys.
        do_augment: Whether to apply random rotation.
        do_normalize: Whether to apply SUV window normalization.
        Anscobe_normalize: Whether to apply Anscombe transform.
        minmax_normalize: Whether to apply min-max normalization.
        lmdb_zfill: Zero-fill width for LMDB keys.
        SUV_window_threshold: SUV window threshold for normalization.
    """

    def __init__(
        self,
        dose_paths: dict,
        prior_path=None,
        projection=None,
        image_size=256,
        original_resolution=256,
        do_augment=False,
        do_normalize=False,
        Anscobe_normalize=False,
        minmax_normalize=False,
        lmdb_zfill=6,
        SUV_window_threshold=4.0,
        **kwargs,
    ):
        if 'full' not in dose_paths:
            raise ValueError("dose_paths must contain a 'full' key")

        self.original_resolution = original_resolution
        self.image_size = image_size
        self.projection = projection

        self.data = {}
        self.dose_keys = []
        self.length = None

        for name, path in dose_paths.items():
            if path is None:
                continue
            db = BaseLMDB(path, original_resolution, zfill=lmdb_zfill)
            if self.length is None:
                self.length = len(db)
            else:
                assert len(db) == self.length, \
                    f"'{name}' length ({len(db)}) != 'full' length ({self.length})"
            self.data[name] = db
            self.dose_keys.append(name)

        self.has_prior = False
        if prior_path is not None:
            db = BaseLMDB(prior_path, original_resolution, zfill=lmdb_zfill)
            assert len(db) == self.length, \
                f"prior length ({len(db)}) != dose data length ({self.length})"
            self.data['prior'] = db
            self.has_prior = True

        self.all_keys = list(self.dose_keys)
        if self.has_prior:
            self.all_keys.append('prior')

        self.SUV_window_threshold = SUV_window_threshold

        transform = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
        if do_augment:
            transform.append(transforms.RandomRotation(15))
        if do_normalize:
            transform.append(transforms.Lambda(
                lambda x: torch.clamp(
                    x, max=self.SUV_window_threshold
                ) / self.SUV_window_threshold))
        if Anscobe_normalize:
            transform.append(transforms.Lambda(
                lambda x: 2 * torch.sqrt(x + 3 / 8)))
        if minmax_normalize:
            transform.append(transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())))
        self.transform = transforms.Compose(transform)

        prior_transform = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
        if prior_path is not None and "CT" in prior_path.upper():
            WL, WW = -600, 1500
            L, U = WL - WW / 2, WL + WW / 2
            prior_transform.extend([
                transforms.Lambda(lambda x: torch.clamp(x, L, U)),
                transforms.Lambda(lambda x: (x - L) / (U - L)),
            ])
        elif prior_path is not None and any(
            tag in prior_path.upper() for tag in ("MR", "T1", "T2")
        ):
            prior_transform.append(transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())))
        self.prior_transform = transforms.Compose(prior_transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        result = {}
        for key in self.all_keys:
            raw = self.data[key][index]
            if raw is None:
                continue
            if key == 'prior':
                result[key] = self.prior_transform(raw)
            else:
                result[key] = self.transform(raw)
        return result


class SimulatedDataset(Dataset):
    """Simulated PET dataset (e.g. BrainWeb) with optional structural prior.

    Returns a dict with ``'full'`` and optionally ``'prior'`` keys.

    Args:
        pet_path: Path to the PET LMDB database.
        prior_path: Path to the structural prior LMDB database.
        projection: Radon projection operator (optional).
        image_size: Target image size after resizing.
        original_resolution: Resolution stored in LMDB keys.
        do_augment: Whether to apply random rotation.
        do_normalize: Whether to apply SUV window normalization.
        W: SUV window threshold.
        lmdb_zfill: Zero-fill width for LMDB keys.
        load_keys: List of keys to load (e.g., ['full'], ['full', 'prior']).
    """

    def __init__(
        self,
        pet_path='datasets/brainweb_pet.lmdb',
        prior_path='datasets/brainweb_mr.lmdb',
        projection=None,
        image_size=256,
        original_resolution=256,
        do_augment=False,
        do_normalize=False,
        Anscobe_normalize=False,
        minmax_normalize=False,
        lmdb_zfill=6,
        W=4.0,
        load_keys=None,
        **kwargs,
    ):
        self.original_resolution = original_resolution

        if load_keys is None:
            load_keys = []
            if pet_path is not None:
                load_keys.append('full')
            if prior_path is not None:
                load_keys.append('prior')

        if 'full' not in load_keys:
            if pet_path is None:
                raise ValueError(
                    "load_keys must include 'full', or provide pet_path")
            load_keys.insert(0, 'full')

        self.data_pet = None
        self.data_prior = None

        if 'full' in load_keys:
            if pet_path is None:
                raise ValueError(
                    "pet_path is required when loading 'full' key")
            self.data_pet = BaseLMDB(
                pet_path, original_resolution, zfill=lmdb_zfill)
            self.length = len(self.data_pet)

        if 'prior' in load_keys:
            if prior_path is None:
                raise ValueError(
                    "prior_path is required when loading 'prior' key")
            self.data_prior = BaseLMDB(
                prior_path, original_resolution, zfill=lmdb_zfill)
            assert len(self.data_prior) == self.length, \
                "PET and prior data length mismatch"

        self.image_size = image_size
        self.SUV_window_threshold = W

        transform = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
        if do_augment:
            transform.append(transforms.RandomRotation(15))
        if do_normalize:
            transform.append(transforms.Lambda(
                lambda x: torch.clamp(
                    x, max=self.SUV_window_threshold
                ) / self.SUV_window_threshold))
        if Anscobe_normalize:
            transform.append(transforms.Lambda(
                lambda x: 2 * torch.sqrt(x + 3 / 8)))
        if minmax_normalize:
            transform.append(transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())))
        self.transform = transforms.Compose(transform)

        prior_transform = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
        if prior_path is not None and "CT" in prior_path.upper():
            WL, WW = -600, 1500
            L, U = WL - WW / 2, WL + WW / 2
            prior_transform.extend([
                transforms.Lambda(lambda x: torch.clamp(x, L, U)),
                transforms.Lambda(lambda x: (x - L) / (U - L)),
            ])
        elif prior_path is not None and any(
            tag in prior_path.upper() for tag in ("MR", "T1", "T2")
        ):
            prior_transform.append(transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())))
        self.prior_transform = transforms.Compose(prior_transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        result = {}
        if self.data_pet is not None:
            pet = self.data_pet[index]
            if self.transform is not None:
                pet = self.transform(pet)
            result['full'] = pet
        if self.data_prior is not None:
            prior = self.data_prior[index]
            if self.prior_transform is not None:
                prior = self.prior_transform(prior)
            result['prior'] = prior
        return result
