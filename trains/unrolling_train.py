"""
Training and validation loops for the unrolling-based PET reconstruction.
"""
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
from torch import nn
from pytorch_msssim import SSIM

from utils.utils import get_mean_ssim, get_mean_psnr, get_mean_rmse, Fourier_Utils
from utils.criterion_registry import DEEP_MSE_loss, WEIGHTED_MSE_loss, WSTL_loss

torch.autograd.set_detect_anomaly(True)

V_MAX = 1
V_MIN = 0


def mlem_reconstruction(sino, pbeam, num_iters=20, eps=1e-3,
                        ratio_max=2.0, alpha=1.0, early_stop_tol=1e-4):
    device = sino.device
    B, C, N_ang, N_bins = sino.shape
    H, W = pbeam.volume.height, pbeam.volume.width
    x = torch.ones((B, 1, H, W), device=device)
    ones_sino = torch.ones_like(sino)
    S = torch.clamp(pbeam.backprojection(ones_sino), min=eps)
    prev_x = x.clone()
    for _ in range(num_iters):
        proj_est = torch.clamp(pbeam.forward(x), min=1e-6)
        ratio = torch.clamp(sino / proj_est, min=1e-6, max=ratio_max)
        ratio[ratio != ratio] = 0
        back = pbeam.backprojection(ratio)
        x_update = x * back / S
        x = x.pow(1 - alpha) * x_update.pow(alpha) if alpha != 1.0 else x_update
        x = torch.clamp(x, min=0.0)
        diff = torch.norm(x - prev_x) / (torch.norm(prev_x) + 1e-9)
        if diff < early_stop_tol:
            break
        prev_x = x.clone()
    return x


def fbp_reconstruction(sino, pbeam):
    sino = pbeam.filter_sinogram(sino)
    return nn.ReLU()(pbeam.backward(sino))


def lowdose_simulate(count=2e5, pbeam=None, full_dose=None, thresh=None):
    if thresh is not None:
        full_dose = torch.clip(full_dose, min=0, max=thresh) / thresh
    proj = pbeam.forward(full_dose)
    mul_factor = torch.ones_like(proj)
    mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
    noise = (torch.ones_like(proj)
             * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True)
             * 0.2)
    sinogram = mul_factor * proj + noise
    cs = count / (1e-9 + torch.sum(sinogram, dim=(-1, -2), keepdim=True))
    sinogram = sinogram * cs
    mul_factor = mul_factor * cs
    noise = noise * cs
    x = torch.poisson(sinogram)
    sino = nn.ReLU()((x - noise) / mul_factor)
    return sino


def unrolling_train(train_loader, model, criterion, optimizer, epoch,
                    device, p, pbeam=None):
    start_time = time.time()
    loss_record, ssim_record, psnr_record, rmse_record = [], [], [], []

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train()
    for batch_idx, sample in progress_bar:
        data = sample
        keys = data.keys()

        if 'prior' in keys:
            prior = data['prior'].to(device).float()
        if 'full' in keys:
            full = data['full'].to(device).float()

        optimizer.zero_grad()

        assert 'simulate_data' in p or 'data_selection' in p
        if p.get('simulate_data', False):
            sino = lowdose_simulate(
                count=float(p['count']), pbeam=pbeam, full_dose=full)
            input = sino
        else:
            dose_key = p.get('data_selection', 'low')
            assert dose_key in keys, \
                f"data_selection='{dose_key}' not found in data keys {list(keys)}"
            selected = data[dose_key].to(device).float()
            sino = pbeam.forward(selected)
            input = sino

        if p.get('post_process', False):
            input = mlem_reconstruction(sino, pbeam)

        if p['use_prior']:
            if 'prior' not in keys:
                prior = input
            outputs, diff = model(input, prior)
        else:
            outputs, diff = model(input)

        predx = outputs[-1] if isinstance(outputs, list) else outputs

        ssim_module = SSIM(
            data_range=full.max(), size_average=True, channel=1)

        loss_pred = torch.tensor(0.0, device=predx.device)
        amp_loss = torch.tensor(0.0, device=predx.device)
        phase_loss = torch.tensor(0.0, device=predx.device)
        f_loss = torch.tensor(0.0, device=predx.device)
        loss_sim = torch.tensor(0.0, device=predx.device)
        loss_sinogram = torch.tensor(0.0, device=predx.device)
        inner_loss_pred = torch.tensor(0.0, device=predx.device)
        deep_mse_loss = torch.tensor(0.0, device=predx.device)
        wstl_loss = torch.tensor(0.0, device=predx.device)
        weighted_deep_mse_loss = torch.tensor(0.0, device=predx.device)

        for key, value in p['loss_type'].items():
            weight = float(value)
            if weight == 0.0:
                continue
            if key == 'loss_pred':
                loss_pred = weight * criterion(
                    target=full.float(), input=predx)
            elif key == 'loss_phase_amp':
                p_loss, a_loss = Fourier_Utils.get_phase_amp_loss(predx, full)
                phase_loss += weight * p_loss
                amp_loss += weight * a_loss
            elif key == 'loss_fourier':
                f_loss += weight * Fourier_Utils.get_fourier_loss(predx, full)
            elif key == 'loss_ssim':
                loss_sim += weight * (1 - ssim_module(full.float(), predx))
            elif key == 'loss_sino':
                loss_sinogram += weight * criterion(
                    target=diff, input=torch.zeros_like(diff))
            elif key == 'inner_loss_pred':
                losses = [criterion(target=full.float(), input=o)
                          for o in outputs]
                inner_loss_pred += weight * torch.mean(torch.stack(losses))
            elif key == 'loss_deep_mse':
                deep_mse_loss += weight * DEEP_MSE_loss(outputs, full)
            elif key == 'loss_weighted_mse':
                weighted_deep_mse_loss += weight * WEIGHTED_MSE_loss(
                    outputs, full)
            elif key == 'loss_wstl':
                wstl_loss += weight * WSTL_loss(outputs)

        loss = (loss_pred + amp_loss + phase_loss + f_loss + loss_sim
                + loss_sinogram + inner_loss_pred + deep_mse_loss
                + weighted_deep_mse_loss + wstl_loss)
        loss.backward()
        optimizer.step()

        loss_record.append(loss.item())
        with torch.no_grad():
            np.seterr(divide='ignore', invalid='ignore')
            pred_img = np.squeeze(predx.cpu().detach().numpy(), axis=1)
            ref = np.squeeze(full.cpu().detach().numpy(), axis=1)

            ssim_record.append(get_mean_ssim(ref, pred_img))
            psnr_record.append(get_mean_psnr(ref, pred_img))
            rmse_record.append(get_mean_rmse(ref, pred_img))

            postfix_dict = {}
            for name, val in [
                ('L_pred', loss_pred), ('L_phase', phase_loss),
                ('L_amp', amp_loss), ('L_fourier', f_loss),
                ('L_ssim', loss_sim), ('L_sino', loss_sinogram),
                ('L_inner', inner_loss_pred), ('L_deep_mse', deep_mse_loss),
                ('L_wstl', wstl_loss),
                ('L_weighted_mse', weighted_deep_mse_loss),
            ]:
                if val.item() != 0.0:
                    postfix_dict[name] = val.item()
            progress_bar.set_postfix(**postfix_dict)

            if batch_idx % (len(train_loader) - 1) == 0 and batch_idx != 0:
                end_time = time.time()
                print(
                    f'Train Epoch: {epoch} '
                    f'[batch:{batch_idx}/{len(train_loader)} '
                    f'({batch_idx * 100.0 / len(train_loader):.0f})%]\t'
                    f'Loss/SSIM/PSNR:'
                    f'{np.mean(loss_record):.6f} / '
                    f'{np.mean(ssim_record):.6f} / '
                    f'{np.mean(psnr_record):.6f} \t'
                    f'Using {end_time - start_time:.3f}s')
                start_time = time.time()

                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning)
                fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
                images = [
                    np.squeeze(ref[0]),
                    np.squeeze(pred_img[0]),
                    np.abs(np.squeeze(pred_img[0] - ref[0])),
                ]
                titles = ['Reference', 'Prediction', 'Error']
                for ax, img, title in zip(axes, images, titles):
                    im = ax.imshow(img, cmap='jet', vmin=V_MIN, vmax=V_MAX)
                    ax.set_title(title, fontsize=12, pad=6)
                    ax.axis('off')
                    cbar = fig.colorbar(
                        im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_ticks([V_MIN, 0.5, V_MAX])

                fig.suptitle(f'Epoch = {epoch}', fontsize=14, y=1.02)
                plt.tight_layout(rect=[0, 0, 1, 0.88])
                plot_path = os.path.join(
                    p['figures_base'],
                    f'img_plot_epoch_{epoch}_{int(time.time())}.png')
                fig.savefig(plot_path)
                plt.close(fig)

    return (np.mean(loss_record), np.mean(ssim_record),
            np.mean(psnr_record), np.mean(rmse_record))


def unrolling_val(val_loader, model, criterion, optimizer, epoch,
                  device, p, pbeam=None):
    model.eval()
    start_time = time.time()
    ssim_record, psnr_record, rmse_record = [], [], []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for batch_idx, sample in progress_bar:
            data = sample
            keys = data.keys()
            if 'prior' in keys:
                prior = data['prior'].to(device).float()
            if 'full' in keys:
                full = data['full'].to(device).float()

            if p.get('simulate_data', False):
                sino = lowdose_simulate(
                    count=float(p['count']), pbeam=pbeam, full_dose=full)
                input = sino
            else:
                dose_key = p.get('data_selection', 'low')
                assert dose_key in keys, \
                    f"data_selection='{dose_key}' not found in data keys {list(keys)}"
                selected = data[dose_key].to(device).float()
                sino = pbeam.forward(selected)
                input = sino

            if p.get('post_process', False):
                input = mlem_reconstruction(sino, pbeam)

            if p['use_prior']:
                if 'prior' not in keys:
                    prior = input
                outputs, diff = model(input, prior)
            else:
                outputs, diff = model(input)

            predx = outputs[-1] if isinstance(outputs, list) else outputs

            np.seterr(divide='ignore', invalid='ignore')
            pred_img = np.squeeze(predx.cpu().detach().numpy(), axis=1)
            ref = np.squeeze(full.cpu().detach().numpy(), axis=1)

            ssim_record.append(get_mean_ssim(ref, pred_img, data_range=1))
            psnr_record.append(get_mean_psnr(ref, pred_img, data_range=1))
            rmse_record.append(get_mean_rmse(ref, pred_img))
            progress_bar.set_postfix(
                Ssim=ssim_record[-1], Psnr=psnr_record[-1],
                Rmse=rmse_record[-1])

            if batch_idx == int(0.5 * len(val_loader)):
                end_time = time.time()
                print(
                    f'Val Epoch: {epoch} '
                    f'[batch:{batch_idx}/{len(val_loader)} '
                    f'({batch_idx * 100.0 / len(val_loader):.0f})%]\t'
                    f'SSIM/PSNR: '
                    f'{np.mean(ssim_record):.6f} / '
                    f'{np.mean(psnr_record):.6f} \t'
                    f'Using {end_time - start_time:.3f}s')
                start_time = time.time()

                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning)
                fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
                rand_idx = np.random.randint(0, len(ref))
                images = [
                    np.squeeze(ref[rand_idx]),
                    np.squeeze(pred_img[rand_idx]),
                    np.abs(np.squeeze(
                        pred_img[rand_idx] - ref[rand_idx])),
                ]
                titles = ['Reference', 'Prediction', 'Error']
                for ax, img, title in zip(axes, images, titles):
                    im = ax.imshow(
                        img, cmap='jet', vmin=V_MIN, vmax=V_MAX)
                    ax.set_title(title, fontsize=12, pad=6)
                    ax.axis('off')
                    cbar = fig.colorbar(
                        im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_ticks([V_MIN, 0.5, V_MAX])

                fig.suptitle(f'Epoch = {epoch}', fontsize=14, y=1.02)
                plt.tight_layout(rect=[0, 0, 1, 0.88])
                plot_path = os.path.join(
                    p['figures_base'],
                    f'val_img_plot_epoch_{epoch}_{int(time.time())}.png')
                fig.savefig(plot_path)
                plt.close(fig)

    model.train()
    return (np.mean(ssim_record), np.mean(psnr_record),
            np.mean(rmse_record))
