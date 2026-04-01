"""
Spectral Convolution Module (SCM)

Combines local spatial CNN features with spectral (FNO-style) processing
via State Space Duality blocks operating in the frequency domain.
"""
import torch
import torch.nn as nn
import torch.fft
import math

try:
    from .efficientViM_utils import LayerNorm1D
    from .efficientViM import FreqHSMSSD
except ImportError:
    from efficientViM_utils import LayerNorm1D
    from efficientViM import FreqHSMSSD


class LocalCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.pw_conv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1)
        self.net_k3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, groups=hidden_channels),
            nn.GELU()
        )
        self.net_k5 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5,
                      padding=2, groups=hidden_channels),
            nn.GELU()
        )
        self.pw_conv_out = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.pw_conv(x)
        x_k3, x_k5 = x.chunk(2, dim=1)
        x_k3 = self.net_k3(x_k3)
        x_k5 = self.net_k5(x_k5)
        x = torch.cat([x_k3, x_k5], dim=1)
        return self.pw_conv_out(x)


class SpectralConv2d_ssd(nn.Module):
    """Spectral convolution via State Space Duality in the frequency domain."""

    def __init__(self, in_ch, out_ch, mode1=16, mode2=16, ssd_state_dim=64):
        super().__init__()
        self.in_ch = in_ch
        self.mode1, self.mode2 = mode1, mode2
        self.norm = LayerNorm1D(in_ch * 2)
        self.freq_ssd = FreqHSMSSD(in_ch * 2, ssd_expand=1, state_dim=ssd_state_dim * 2)

    def forward(self, x: torch.Tensor, h_prev=None):
        xf = torch.fft.rfft2(x, norm='ortho')
        Hf, Wf = xf.shape[-2:]
        sqrt_n = math.sqrt(x.shape[-2] * x.shape[-1])

        freq_in = torch.cat([xf.real.flatten(2), xf.imag.flatten(2)], dim=1)
        freq_in = self.norm(freq_in) / sqrt_n
        freq_out, h_freq = self.freq_ssd(
            freq_in, Hf, Wf, h_prev if h_prev is not None else None)

        freq_out = freq_out * sqrt_n
        real_out, imag_out = freq_out.chunk(2, dim=1)
        ft_out = torch.complex(real_out, imag_out)
        y = torch.fft.irfft2(ft_out, s=x.shape[-2:], norm='ortho')
        return y, h_freq


class FNOBlock(nn.Module):
    def __init__(self, hidden_channels, mode1, mode2, ssd_state_dim):
        super().__init__()
        self.pw = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.fno = SpectralConv2d_ssd(
            hidden_channels, hidden_channels, mode1, mode2, ssd_state_dim)
        self.post = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
        )

    def forward(self, x, alpha, h=None):
        x1 = self.pw(x)
        x2, h = self.fno(x, h)
        x2 = x2 + self.post(x2)
        out = (1 - alpha) * x1 + alpha * x2
        return out, h


class SCM_module(nn.Module):
    """
    Spectral Convolution Module.

    Combines local spatial features (LocalCNN) with spectral processing
    (FNO blocks with learnable spatial-vs-frequency gating).
    """

    def __init__(self, in_channels, hidden_channels, num_fno_layers=4,
                 mode1=16, mode2=16, ssd_state_dim=64):
        super().__init__()
        self.local = LocalCNN(in_channels * 3, hidden_channels)
        self.skip_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.pw_conv = nn.Conv2d(
            in_channels * 3 + hidden_channels, hidden_channels, kernel_size=1)
        self.gate_alpha = nn.Parameter(torch.zeros(num_fno_layers, 1, 1, 1))
        self.blocks = nn.ModuleList([
            FNOBlock(hidden_channels, mode1, mode2, ssd_state_dim)
            for _ in range(num_fno_layers)
        ])
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x, h=None):
        local_feat = self.local(x)
        spatial_feat = self.pw_conv(torch.cat([x, local_feat], dim=1))
        x = spatial_feat
        alphas = torch.sigmoid(self.gate_alpha)

        for alpha, blk in zip(alphas, self.blocks):
            x, h = blk(x, alpha, h)
        x = self.to_out(x) + self.skip_conv(spatial_feat)
        return x, h
