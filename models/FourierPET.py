"""
FourierPET: ADMM-based unrolling network for PET image reconstruction.

The network alternates between:
  - X-update (SCM: spectral convolution module)
  - Z-update (APCM: amplitude-phase correction module)
  - Dual variable update
"""
import torch
import torch.nn as nn
import numpy as np

try:
    from .SCM import SCM_module
    from .APCM import APCM_module
except ImportError:
    from SCM import SCM_module
    from APCM import APCM_module


class Fourier_Utils:
    @staticmethod
    def fft2d(x):
        return torch.fft.fftshift(torch.fft.fft2(x))

    @staticmethod
    def ifft2d(x):
        return torch.fft.ifft2(torch.fft.ifftshift(x))

    @staticmethod
    def get_amplitude(x):
        return torch.abs(Fourier_Utils.fft2d(x))

    @staticmethod
    def get_phase(x):
        return torch.angle(Fourier_Utils.fft2d(x))

    @staticmethod
    def normalize_amplitude(amp):
        amp_log = torch.log(amp + 1)
        min_v = amp_log.amin(dim=[2, 3], keepdim=True)
        max_v = amp_log.amax(dim=[2, 3], keepdim=True)
        return (amp_log - min_v) / (max_v - min_v + 1e-8)

    @staticmethod
    def phase_to_sincos(phase):
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    @staticmethod
    def sincos_to_phase(sincos):
        C2 = sincos.shape[1]
        assert C2 % 2 == 0, "Channel dim must be even"
        C = C2 // 2
        return torch.atan2(sincos[:, :C], sincos[:, C:])

    @staticmethod
    def get_phase_and_amp(x):
        fft = Fourier_Utils.fft2d(x)
        amp = torch.abs(fft)
        amp_norm = Fourier_Utils.normalize_amplitude(amp)
        phase = torch.angle(fft)
        phase_sincos = Fourier_Utils.phase_to_sincos(phase)
        return phase_sincos, amp_norm

    @staticmethod
    def get_phase_amp_loss(pred, target):
        pred_phase, pred_amp = Fourier_Utils.get_phase_and_amp(pred)
        target_phase, target_amp = Fourier_Utils.get_phase_and_amp(target)
        phase_loss = torch.mean(torch.abs(pred_phase - target_phase))
        amp_loss = torch.mean(torch.abs(pred_amp - target_amp))
        return phase_loss, amp_loss

    @staticmethod
    def get_fourier_loss(pred, target):
        pred_f = Fourier_Utils.fft2d(pred)
        target_f = Fourier_Utils.fft2d(target)
        return torch.mean(torch.abs(pred_f - target_f))


class XUpdateBlock(nn.Module):
    """X-update step: spectral convolution on concatenated inputs."""

    def __init__(self, pbeam, inner_iter=10, hidden_dim=16,
                 ssd_state_dim=64, mode1=16, mode2=16):
        super().__init__()
        self.log_rho = nn.Parameter(torch.log(torch.tensor(0.003)))
        self.inner_iter = inner_iter
        self.pbeam = pbeam
        self.FNO_net = SCM_module(
            in_channels=1, num_fno_layers=inner_iter,
            mode1=mode1, mode2=mode2,
            hidden_channels=hidden_dim, ssd_state_dim=ssd_state_dim)

    def forward(self, ins):
        x_temp, sino, z_prev, u_prev, h_prev = ins
        ATsino = self.pbeam.backward(sino) / self.pbeam.angles.shape[0]
        input = torch.cat((x_temp, z_prev - u_prev, ATsino), dim=1)
        x_proposed, h_proposed = self.FNO_net(input, h_prev)
        x1 = x_temp + x_proposed
        return (x1, sino, z_prev, u_prev, h_proposed)


class ZUpdateBlock(nn.Module):
    """Z-update step: wavelet-Fourier amplitude-phase correction."""

    def __init__(self, wave='haar', J=1, spatial_dilations=(1, 2, 4, 8),
                 ffn_dim=32):
        super().__init__()
        self.wavelet_dft = APCM_module(
            in_channels=1, wave=wave, J=J,
            spatial_dilations=spatial_dilations, ffn_dim=ffn_dim)

    def forward(self, ins, prior=None):
        x, y, z_prev, u_prev = ins
        z_proposed = self.wavelet_dft(x + u_prev)
        return (x, y, z_proposed, u_prev)


class InterBlock(nn.Module):
    """Single ADMM iteration block combining X-update, Z-update, and dual update."""

    def __init__(self, inner_iter, pbeam, hidden_dim=16, ssd_state_dim=64,
                 wave='haar', J=1, spatial_dilations=(1, 2, 4, 8),
                 ffn_dim=32, mode1=16, mode2=16):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(0.001))
        self.layers_up_x = nn.ModuleList([
            XUpdateBlock(pbeam, inner_iter, hidden_dim=hidden_dim,
                         ssd_state_dim=ssd_state_dim, mode1=mode1, mode2=mode2)
        ])
        self.layers_up_z = nn.ModuleList([
            ZUpdateBlock(wave=wave, J=J,
                         spatial_dilations=spatial_dilations, ffn_dim=ffn_dim)
            for _ in range(inner_iter)
        ])

    def forward(self, ins, prior=None):
        x, y, z, b, h = ins
        for layer in self.layers_up_x:
            x, y, z, b, h = layer((x, y, z, b, h))
        for layer in self.layers_up_z:
            x, y, z, b = layer((x, y, z, b), prior)
        b = b + self.eta * (x - z)
        return x, y, z, b, h


class FourierPET(nn.Module):
    """
    FourierPET: ADMM-unrolled network for PET reconstruction.

    Args:
        admm_iter: Number of ADMM outer iterations (unrolled blocks).
        inner_iter: Number of inner iterations per ADMM block.
        radon: Parallel beam projection operator.
        hidden_dim: Hidden channel dimension for SCM.
        ssd_state_dim: State dimension for SSD blocks.
        wave: Wavelet type for DWT in APCM.
        J: Number of DWT decomposition levels.
        spatial_dilations: Dilation rates for spatial convolutions in APCM.
        ffn_dim: FFN expansion dimension in APCM.
        mode1, mode2: Fourier modes for spectral convolution.
    """

    def __init__(self, admm_iter, inner_iter, radon, hidden_dim=16,
                 ssd_state_dim=64, wave='haar', J=1,
                 spatial_dilations=(1, 2, 4, 8), ffn_dim=32,
                 mode1=16, mode2=16):
        super().__init__()
        self.admm_iter = admm_iter
        self.inner_iter = inner_iter
        self.radon = radon
        self.layers = nn.ModuleList([
            InterBlock(
                self.inner_iter, radon, hidden_dim=hidden_dim,
                ssd_state_dim=ssd_state_dim, wave=wave, J=J,
                spatial_dilations=spatial_dilations, ffn_dim=ffn_dim,
                mode1=mode1, mode2=mode2)
            for _ in range(self.admm_iter)
        ])

    def forward(self, sinogram, prior=None):
        outputs = []
        x, y, z, b, h = (
            torch.zeros_like(sinogram), sinogram,
            torch.zeros_like(sinogram), torch.zeros_like(sinogram), None)

        for block in self.layers:
            x, y, z, b, h = block((x, y, z, b, h), prior)
            outputs.append(x)

        sinogram_out = self.radon.forward(x)
        return outputs, sinogram_out - sinogram


def create_FourierPET(admm_iter=10, inner_iter=10, pbeam_config=None,
                      hidden_dim=16, ssd_state_dim=64, wave='haar', J=1,
                      spatial_dilations=(1, 2, 4, 8), ffn_dim=32,
                      mode1=16, mode2=16):
    """Factory function to create a FourierPET model with a ParallelBeam projector."""
    from torch_radon import ParallelBeam
    pbeam = ParallelBeam(
        det_count=pbeam_config['del_count'],
        angles=np.linspace(
            pbeam_config['angles'][0] * np.pi,
            pbeam_config['angles'][1] * np.pi,
            pbeam_config['angles'][2]),
        volume=pbeam_config['volume'])
    return FourierPET(
        admm_iter, inner_iter, pbeam,
        hidden_dim=hidden_dim, ssd_state_dim=ssd_state_dim,
        wave=wave, J=J, spatial_dilations=spatial_dilations,
        ffn_dim=ffn_dim, mode1=mode1, mode2=mode2)
