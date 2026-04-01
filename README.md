# FourierPET

Official implementation of **FourierPET**: an ADMM-unrolled deep network for PET image reconstruction that jointly operates in spatial, wavelet, and Fourier domains.

<!-- TODO: Add paper link after publication -->
<!-- > **Paper Title**  
> Author1, Author2, ...  
> *Conference/Journal, Year*  
> [[Paper]](link) [[arXiv]](link) -->

## Overview

FourierPET is a learnable ADMM-based unrolling framework for PET image reconstruction. Each ADMM iteration consists of:

- **X-update** via the **Spectral Convolution Module (SCM)**: combines local spatial features (depthwise CNN) with global spectral features (FNO-style blocks using State Space Duality in the frequency domain).
- **Z-update** via the **Amplitude-Phase Correction Module (APCM)**: decomposes features using DWT, then processes each subband in both spatial and Fourier domains with separate amplitude and phase correction branches.
- **Dual variable update** with a learnable step size.

## Project Structure

```
FourierPET/
├── train.py                  # Main training script
├── configs/
│   ├── env.yml               # Environment config (output directory)
│   └── FourierPET_3_2.yml    # Experiment config
├── models/
│   ├── FourierPET.py          # FourierPET network (ADMM unrolling)
│   ├── SCM.py                 # Spectral Convolution Module
│   ├── APCM.py                # Amplitude-Phase Correction Module
│   ├── efficientViM.py        # HSM-SSD blocks
│   └── efficientViM_utils.py  # Layer utilities
├── data/
│   └── pet_data.py            # LMDB-based PET data loader
├── trains/
│   └── unrolling_train.py     # Training and validation loops
└── utils/
    ├── config.py              # Config loading
    ├── common_config.py       # Object creation factories
    ├── dataset_registry.py    # Dataset registry
    ├── model_registry.py      # Model registry
    ├── criterion_registry.py  # Loss function registry
    ├── optimizer_registry.py  # Optimizer registry
    ├── projection_simulation.py  # Radon projection utilities
    └── utils.py               # Metrics (SSIM, PSNR, RMSE) and Fourier utils
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/FourierPET.git
cd FourierPET
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.12
- [torch-radon](https://github.com/matteo-ronchetti/torch-radon) (for parallel beam projection)
- [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) (for DWT/iDWT)
- See `requirements.txt` for the full list.

## Data Preparation

This project uses LMDB-format datasets. Two dataset modes are supported:

### Simulated Data (e.g. BrainWeb)

```yaml
train_db_name: Simulated_data
val_db_name: Simulated_data
pet_path: /path/to/brainweb_pet.lmdb
prior_path: /path/to/brainweb_mr.lmdb
```

### Real Clinical Data (multi-dose)

Dose levels are configured via a `dose_paths` dict — keys are arbitrary names
(e.g. scan durations), and `full` is required as the reference target.

```yaml
train_db_name: Real_clinical_data
val_db_name: Real_clinical_data

dose_paths:
    full: /path/to/train_full_dose.lmdb
    5min: /path/to/train_5min.lmdb
    1min: /path/to/train_1min.lmdb
    6s:   /path/to/train_6s.lmdb
prior_path: /path/to/train_CT.lmdb

val_dose_paths:
    full: /path/to/test_full_dose.lmdb
    5min: /path/to/test_5min.lmdb
    1min: /path/to/test_1min.lmdb
    6s:   /path/to/test_6s.lmdb
val_prior_path: /path/to/test_CT.lmdb
```

## Training

```bash
python train.py --config_exp configs/FourierPET_3_2.yml
```

Training outputs (checkpoints, logs, figures) are saved to `outputs/FourierPET_3_2/`.

## Evaluation

```bash
# Single checkpoint
python test.py --config_exp configs/FourierPET_3_2.yml \
               --checkpoint outputs/FourierPET_3_2/checkpoint.pth.tar

# LOOCV — evaluate all folds
python test.py --config_exp configs/FourierPET_3_2.yml --loocv

# Evaluate a specific fold
python test.py --config_exp configs/FourierPET_3_2.yml --loocv_fold 3

# Save reconstructed images
python test.py --config_exp configs/FourierPET_3_2.yml \
               --checkpoint outputs/.../checkpoint.pth.tar --save_images
```

Results (JSON metrics and optional image grids) are saved to `outputs/<exp_name>/test_results/`.

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backbone_kwargs.admm_iter` | Number of ADMM iterations | 3 |
| `backbone_kwargs.inner_iter` | Inner iterations per ADMM block | 2 |
| `backbone_kwargs.hidden_dim` | Hidden channels for SCM | 48 |
| `backbone_kwargs.ssd_state_dim` | State dimension for SSD | 48 |
| `backbone_kwargs.wave` | Wavelet type for APCM | haar |
| `count` | Simulated photon count | 2e5 |
| `epochs` | Training epochs | 100 |
| `batch_size` | Batch size | 16 |
| `LOOCV` | Enable leave-one-out cross-validation | True |

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025fourierpet,
  title={FourierPET: ...},
  author={Zhang, Zheng and ...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
