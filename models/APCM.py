"""
Amplitude-Phase Correction Module (APCM)

Decomposes input via DWT, processes each subband in both spatial and
Fourier domains (amplitude + phase branches), then reconstructs via iDWT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

try:
    from .efficientViM_utils import FFN
except ImportError:
    from efficientViM_utils import FFN


class Fourier_Utils:
    @staticmethod
    def fft2d(x):
        return torch.fft.fftshift(torch.fft.fft2(x))

    @staticmethod
    def fourier_to_phase_amp(f):
        return torch.angle(f), torch.abs(f)

    @staticmethod
    def ifft2d(x):
        return torch.fft.ifft2(torch.fft.ifftshift(x))

    @staticmethod
    def phase_to_sincos(phase):
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    @staticmethod
    def sincos_to_phase(sincos):
        C2 = sincos.shape[1]
        assert C2 % 2 == 0, "Channel dim must be even"
        C = C2 // 2
        sin = sincos[:, :C, :, :]
        cos = sincos[:, C:, :, :]
        return torch.atan2(sin, cos)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class APCM_module(nn.Module):
    """
    Amplitude-Phase Correction Module.

    Pipeline per forward pass:
      1) DWT decomposition into LL + detail subbands
      2) Spatial branch: grouped dilated depthwise conv
      3) FFT → amplitude branch (1×1 conv + LL-specific FFN)
      4) FFT → phase branch (1×1 conv + HH-specific FFN + cross-band FFN)
      5) Gated spectral fusion + iFFT
      6) iDWT reconstruction
    """

    def __init__(
        self, in_channels, wave='haar', J=1,
        spatial_dilations=(1, 2, 4, 8), ffn_dim=32, expand=2
    ):
        super().__init__()
        self.in_ch = in_channels
        self.J = J
        self.expand = expand
        self.dwt = DWTForward(J=J, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        num_subs = 1 + 3 * J
        C_total = in_channels * num_subs
        D = len(spatial_dilations)
        self.D = D

        # Spatial branch
        self.spatial_pw_in = nn.Conv2d(
            C_total, C_total * D, kernel_size=1, groups=num_subs, bias=False)
        self.spatial_dilated = nn.ModuleList([
            nn.Conv2d(C_total, C_total, 3, padding=d, dilation=d,
                      groups=C_total, bias=False)
            for d in spatial_dilations
        ])
        self.spatial_pw_out = nn.Conv2d(
            C_total * D, C_total, kernel_size=1, groups=num_subs, bias=False)
        self.spatial_gate = nn.Conv2d(
            C_total, C_total, kernel_size=1, groups=num_subs, bias=True)

        # Amplitude branch
        amp_c = in_channels
        amp_total = amp_c * num_subs
        self.amp_conv = nn.Sequential(
            nn.Conv2d(amp_total, amp_total, 1, bias=False),
            nn.BatchNorm2d(amp_total),
            nn.GELU(),
        )
        self.amp_ll_conv = nn.Sequential(
            nn.BatchNorm2d(amp_c),
            nn.Conv2d(amp_c, amp_c * ffn_dim, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(amp_c * ffn_dim, amp_c, 1, bias=True),
        )
        self.amp_gate = nn.Conv2d(
            amp_total, amp_total, kernel_size=1, groups=num_subs, bias=True)

        # Phase branch
        phase_c = in_channels * 2
        phase_total = phase_c * num_subs
        self.phase_conv = nn.Sequential(
            nn.Conv2d(phase_total, phase_total, 1, bias=False),
            nn.BatchNorm2d(phase_total),
            nn.GELU(),
        )
        self.phase_hh_conv = nn.Sequential(
            nn.BatchNorm2d(phase_c),
            nn.Conv2d(phase_c, phase_c * ffn_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(phase_c * ffn_dim, phase_c, 1, bias=False),
        )
        self.phase_gate = nn.Conv2d(
            phase_total, phase_total, kernel_size=1, groups=num_subs, bias=True)
        self.phase_ffn = nn.Sequential(
            nn.Conv2d(num_subs * 2, num_subs * 2 * ffn_dim * expand,
                      kernel_size=1, bias=False),
            nn.GELU(),
            SimpleGate(),
            nn.Conv2d(num_subs * 2 * ffn_dim, num_subs * 2,
                      kernel_size=1, bias=False),
        )

        self.spec_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 1) DWT
        Yl, Yh_list = self.dwt(x)
        subs = [Yl.unsqueeze(2)]
        for Yh in Yh_list:
            subs.append(Yh)
        subs = torch.cat(subs, dim=2)
        B, C, N, H1, W1 = subs.shape
        subs = subs.permute(0, 2, 1, 3, 4).reshape(B, C * N, H1, W1)

        # 2) Spatial branch
        sp = self.spatial_pw_in(subs)
        sps = torch.chunk(sp, self.D, dim=1)
        outs = [conv(sps[i]) for i, conv in enumerate(self.spatial_dilated)]
        sp = torch.cat(outs, dim=1)
        sp = self.spatial_pw_out(sp)
        sub_sp = subs + sp

        # 3) FFT
        spec = Fourier_Utils.fft2d(sub_sp)
        phase, amp = Fourier_Utils.fourier_to_phase_amp(spec)

        # 4) Amplitude branch
        amp_p = self.amp_conv(amp)
        LL_amp = amp_p[:, :self.in_ch, :, :]
        rest_amp = amp_p[:, self.in_ch:, :, :]
        LL_amp = LL_amp + self.amp_ll_conv(LL_amp)
        amp_p = torch.cat([LL_amp, rest_amp], dim=1)
        gate_a = torch.sigmoid(self.amp_gate(amp_p))
        amp_p = amp + gate_a * amp_p

        # 5) Phase branch
        sc = Fourier_Utils.phase_to_sincos(phase)
        sc_proc = self.phase_conv(sc)
        HH_phase = sc_proc[:, self.in_ch * 2 * 3:, :, :]
        rest_phase = sc_proc[:, :self.in_ch * 2 * 3, :, :]
        HH_phase = HH_phase + self.phase_hh_conv(HH_phase)
        sc_proc = torch.cat([HH_phase, rest_phase], dim=1)
        sc_proc = self.phase_ffn(sc_proc)
        gate_p = torch.sigmoid(self.phase_gate(sc_proc))
        sc_p = sc + gate_p * sc_proc
        phase_p = Fourier_Utils.sincos_to_phase(sc_p)

        # 6) Spectral fusion + iFFT
        new_spec = amp_p * torch.exp(1j * phase_p)
        spec_p = spec + torch.sigmoid(self.spec_alpha) * (new_spec - spec)
        out = Fourier_Utils.ifft2d(spec_p).real

        # 7) iDWT
        out = out.reshape(B, N, C, H1, W1).permute(0, 2, 1, 3, 4)
        Yl_p = out[:, :, 0]
        Yh_p = out[:, :, 1:].reshape(B, C, 3, H1, W1)
        return self.idwt((Yl_p, [Yh_p]))
