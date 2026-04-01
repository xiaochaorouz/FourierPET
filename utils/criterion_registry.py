"""
Loss function registry.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class ROI_mse(nn.Module):
    """MSE loss computed only where target != 0."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mse_none = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        loss_map = self.mse_none(input=input, target=target)
        mask = (target != 0).float()
        return (loss_map * mask).sum() / (mask.sum() + self.eps)


def DEEP_MSE_loss(outputs, gt):
    return torch.mean(torch.stack(
        [F.mse_loss(o, gt) for o in outputs]))


def WEIGHTED_MSE_loss(outputs, gt, p=4.0):
    T = len(outputs)
    ws = torch.tensor(
        [((t + 1) / T) ** p for t in range(T)],
        device=outputs[0].device, dtype=outputs[0].dtype)
    ws = ws / ws.sum()
    loss = sum(w * F.mse_loss(out, gt) for w, out in zip(ws, outputs))
    return loss


def WSTL_loss(outputs, alpha=0.5, beta=0.5, epsilon=1e-6):
    """Wavelet-Spectral Trajectory Loss for iterative convergence."""
    device = outputs[0].device
    K = len(outputs)
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)

    spec_loss, ll_loss, hh_loss = 0.0, 0.0, 0.0
    for k in range(1, K):
        Xk = torch.fft.rfft2(outputs[k], norm='ortho')
        Xk1 = torch.fft.rfft2(outputs[k - 1], norm='ortho')
        num = torch.norm(Xk.abs() - Xk1.abs(), p='fro') ** 2
        den = torch.norm(Xk1.abs(), p='fro') ** 2 + epsilon
        spec_loss += num / den

        Yl_k, Yh_k = dwt(outputs[k])
        Yl_k1, Yh_k1 = dwt(outputs[k - 1])
        ll_loss += torch.norm(Yl_k - Yl_k1, p='fro') ** 2
        hh_loss += torch.norm(
            Yh_k[0][..., 2] - Yh_k1[0][..., 2], p='fro') ** 2

    return (spec_loss + alpha * hh_loss + beta * ll_loss) / (K - 1)


CRITERION_MAP = {
    'mse': nn.MSELoss,
    'sl1': nn.SmoothL1Loss,
    'ROI_mse': ROI_mse,
}


def get_criterion(p: Dict):
    criterion_name = p['criterion']
    if criterion_name not in CRITERION_MAP:
        raise ValueError(f'Invalid criterion: {criterion_name}')
    return CRITERION_MAP[criterion_name]()
