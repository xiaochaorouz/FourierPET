"""
FourierPET — Evaluation / Inference script.

Usage:
    # Single model checkpoint
    python test.py --config_exp configs/FourierPET_3_2.yml \
                   --checkpoint outputs/FourierPET_3_2/checkpoint.pth.tar

    # LOOCV — evaluate all folds automatically
    python test.py --config_exp configs/FourierPET_3_2.yml --loocv

    # Save reconstructed images
    python test.py --config_exp configs/FourierPET_3_2.yml \
                   --checkpoint outputs/.../checkpoint.pth.tar --save_images
"""
import argparse
import os
import json
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import cprint

from utils.config import create_config
from utils.common_config import (
    get_val_dataset, get_projection_simulation,
    get_val_transformations, get_model,
    get_val_dataloader, get_val_dataloader_LOOCV,
)
from utils.utils import get_mean_ssim, get_mean_psnr, get_mean_rmse
from trains.unrolling_train import lowdose_simulate, mlem_reconstruction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_args():
    parser = argparse.ArgumentParser(description='FourierPET Evaluation')
    parser.add_argument('--config_exp', required=True,
                        help='Path to experiment config file')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to model checkpoint (.pth / .pth.tar)')
    parser.add_argument('--loocv', action='store_true',
                        help='Run LOOCV evaluation across all folds')
    parser.add_argument('--loocv_fold', type=int, default=None,
                        help='Evaluate only a specific LOOCV fold')
    parser.add_argument('--save_images', action='store_true',
                        help='Save reconstructed images to disk')
    parser.add_argument('--output_dir', default=None,
                        help='Override output directory for results')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use (default: cuda:0)')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        epoch = ckpt.get('epoch', -1)
        best_ssim = ckpt.get('best_ssim', -1)
        cprint(f"Loaded checkpoint (epoch {epoch}, best_ssim {best_ssim:.4f}): {checkpoint_path}", 'green')
    else:
        model.load_state_dict(ckpt)
        cprint(f"Loaded model weights: {checkpoint_path}", 'green')
    return model


def evaluate(model, dataloader, device, p, pbeam):
    """Run evaluation on a dataloader, return per-sample metrics and optional images."""
    model.eval()

    all_ssim, all_psnr, all_rmse = [], [], []
    all_preds, all_refs, all_inputs = [], [], []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc='Evaluating'):
            data = sample
            keys = data.keys()

            full = data['full'].to(device).float()
            prior = data.get('prior')
            if prior is not None:
                prior = prior.to(device).float()

            if p.get('simulate_data', False):
                sino = lowdose_simulate(
                    count=float(p['count']), pbeam=pbeam, full_dose=full)
                inp = sino
            else:
                dose_key = p.get('data_selection', 'low')
                assert dose_key in keys, \
                    f"data_selection='{dose_key}' not found in data keys {list(keys)}"
                selected = data[dose_key].to(device).float()
                sino = pbeam.forward(selected)
                inp = sino

            if p.get('post_process', False):
                inp = mlem_reconstruction(sino, pbeam)

            if p.get('use_prior', False):
                if prior is None:
                    prior = inp
                outputs, diff = model(inp, prior)
            else:
                outputs, diff = model(inp)

            predx = outputs[-1] if isinstance(outputs, list) else outputs

            pred_np = np.squeeze(predx.cpu().numpy(), axis=1)
            ref_np = np.squeeze(full.cpu().numpy(), axis=1)
            inp_np = np.squeeze(inp.cpu().numpy(), axis=1) if inp.dim() == 4 else None

            np.seterr(divide='ignore', invalid='ignore')
            all_ssim.append(get_mean_ssim(ref_np, pred_np, data_range=1))
            all_psnr.append(get_mean_psnr(ref_np, pred_np, data_range=1))
            all_rmse.append(get_mean_rmse(ref_np, pred_np))

            all_preds.append(pred_np)
            all_refs.append(ref_np)
            if inp_np is not None:
                all_inputs.append(inp_np)

    metrics = {
        'ssim': float(np.mean(all_ssim)),
        'psnr': float(np.mean(all_psnr)),
        'rmse': float(np.mean(all_rmse)),
        'ssim_std': float(np.std(all_ssim)),
        'psnr_std': float(np.std(all_psnr)),
        'rmse_std': float(np.std(all_rmse)),
        'num_batches': len(all_ssim),
    }

    images = {
        'preds': np.concatenate(all_preds, axis=0),
        'refs': np.concatenate(all_refs, axis=0),
    }
    if all_inputs:
        images['inputs'] = np.concatenate(all_inputs, axis=0)

    return metrics, images


def save_sample_images(images, save_dir, num_samples=8):
    """Save a grid of sample reconstructions."""
    os.makedirs(save_dir, exist_ok=True)
    n = min(num_samples, len(images['refs']))
    indices = np.linspace(0, len(images['refs']) - 1, n, dtype=int)

    has_input = 'inputs' in images
    ncols = 4 if has_input else 3

    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3, n * 3), dpi=150)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        ref = images['refs'][idx]
        pred = images['preds'][idx]
        err = np.abs(pred - ref)

        col = 0
        if has_input:
            axes[row, col].imshow(images['inputs'][idx], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title('Input' if row == 0 else '', fontsize=10)
            axes[row, col].axis('off')
            col += 1

        axes[row, col].imshow(ref, cmap='hot', vmin=0, vmax=1)
        axes[row, col].set_title('Reference' if row == 0 else '', fontsize=10)
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[row, col + 1].set_title('Prediction' if row == 0 else '', fontsize=10)
        axes[row, col + 1].axis('off')

        im = axes[row, col + 2].imshow(err, cmap='jet', vmin=0, vmax=0.3)
        axes[row, col + 2].set_title('Error' if row == 0 else '', fontsize=10)
        axes[row, col + 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'reconstruction_samples.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    cprint(f"Saved sample images -> {save_path}", 'blue')


def evaluate_single(args, p, device):
    """Evaluate a single checkpoint."""
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = p.get('checkpoint', None)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found. Provide --checkpoint or ensure "
            f"'checkpoint' exists in config output dir.")

    projection = get_projection_simulation(p)
    val_transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, projection, val_transforms)
    val_loader = get_val_dataloader(p, val_dataset)

    model = get_model(p).to(device)
    model = load_checkpoint(model, checkpoint_path, device)

    metrics, images = evaluate(model, val_loader, device, p, projection)

    result_dir = args.output_dir or os.path.join(p.get('output_dir', 'outputs'), 'test_results')
    os.makedirs(result_dir, exist_ok=True)

    cprint(f"\nResults:", 'cyan')
    cprint(f"  SSIM: {metrics['ssim']:.4f} +/- {metrics['ssim_std']:.4f}", 'cyan')
    cprint(f"  PSNR: {metrics['psnr']:.4f} +/- {metrics['psnr_std']:.4f}", 'cyan')
    cprint(f"  RMSE: {metrics['rmse']:.4f} +/- {metrics['rmse_std']:.4f}", 'cyan')

    metrics_path = os.path.join(result_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    cprint(f"Saved metrics -> {metrics_path}", 'blue')

    if args.save_images:
        save_sample_images(images, result_dir)

    return metrics


def evaluate_loocv(args, p, device):
    """Evaluate all LOOCV folds."""
    base_dir = p.get('output_dir', 'outputs')
    num_folds = p.get('LOOCV_num', 20)

    if args.loocv_fold is not None:
        fold_range = [args.loocv_fold]
    else:
        fold_range = range(num_folds)

    projection = get_projection_simulation(p)
    val_transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, projection, val_transforms)

    all_fold_metrics = []

    for fold_idx in fold_range:
        cprint(f"\n{'='*60}", 'yellow')
        cprint(f"Evaluating LOOCV Fold {fold_idx}/{num_folds}", 'yellow')
        cprint(f"{'='*60}", 'yellow')

        fold_dir = os.path.join(base_dir, f"LOOCV_{fold_idx}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")

        best_ckpt = os.path.join(ckpt_dir, f"best_checkpoint_fold{fold_idx}.pth")
        final_model = os.path.join(ckpt_dir, f"final_model_fold{fold_idx}.pth")

        if args.checkpoint:
            ckpt_path = args.checkpoint
        elif os.path.exists(best_ckpt):
            ckpt_path = best_ckpt
        elif os.path.exists(final_model):
            ckpt_path = final_model
        else:
            cprint(f"  [Fold {fold_idx}] No checkpoint found, skipping.", 'red')
            continue

        spc = p.get('slices_per_case', len(val_dataset) // num_folds)
        val_loader = get_val_dataloader_LOOCV(
            p, val_dataset, fold_idx, slices_per_case=spc)

        model = get_model(p).to(device)
        model = load_checkpoint(model, ckpt_path, device)

        metrics, images = evaluate(model, val_loader, device, p, projection)

        cprint(f"  Fold {fold_idx}: SSIM={metrics['ssim']:.4f}, "
               f"PSNR={metrics['psnr']:.4f}, RMSE={metrics['rmse']:.4f}", 'cyan')

        all_fold_metrics.append(metrics)

        if args.save_images:
            fold_result_dir = os.path.join(
                args.output_dir or base_dir, 'test_results', f'fold_{fold_idx}')
            save_sample_images(images, fold_result_dir, num_samples=4)

        del model
        torch.cuda.empty_cache()

    if not all_fold_metrics:
        cprint("No folds were evaluated.", 'red')
        return

    ssim_vals = [m['ssim'] for m in all_fold_metrics]
    psnr_vals = [m['psnr'] for m in all_fold_metrics]
    rmse_vals = [m['rmse'] for m in all_fold_metrics]

    summary = {
        'num_folds': len(all_fold_metrics),
        'ssim_mean': float(np.mean(ssim_vals)),
        'ssim_std': float(np.std(ssim_vals)),
        'psnr_mean': float(np.mean(psnr_vals)),
        'psnr_std': float(np.std(psnr_vals)),
        'rmse_mean': float(np.mean(rmse_vals)),
        'rmse_std': float(np.std(rmse_vals)),
        'per_fold': all_fold_metrics,
    }

    cprint(f"\n{'='*60}", 'magenta')
    cprint(f"LOOCV Summary ({len(all_fold_metrics)} folds):", 'magenta')
    cprint(f"  SSIM: {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}", 'magenta')
    cprint(f"  PSNR: {summary['psnr_mean']:.4f} +/- {summary['psnr_std']:.4f}", 'magenta')
    cprint(f"  RMSE: {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f}", 'magenta')
    cprint(f"{'='*60}", 'magenta')

    result_dir = args.output_dir or os.path.join(base_dir, 'test_results')
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, 'loocv_test_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    cprint(f"Saved LOOCV summary -> {summary_path}", 'blue')

    return summary


def main():
    torch.manual_seed(0)
    args = parse_args()
    p = create_config(args.config_exp)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    cprint(f"Config: {args.config_exp}", 'green')
    cprint(f"Device: {device}", 'green')

    if args.loocv or args.loocv_fold is not None:
        evaluate_loocv(args, p, device)
    else:
        evaluate_single(args, p, device)


if __name__ == "__main__":
    main()
