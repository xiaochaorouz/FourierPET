"""
EfficientViM: Hidden State Mixer – State Space Duality (HSM-SSD) modules.

Provides HSMSSD and FreqHSMSSD blocks used by the Spectral Convolution Module.
"""
from timm.models.vision_transformer import trunc_normal_
import torch.nn as nn
import torch

try:
    from .efficientViM_utils import (
        LayerNorm1D, LayerNorm2D, ConvLayer1D, ConvLayer2D, FFN)
except ImportError:
    from efficientViM_utils import (
        LayerNorm1D, LayerNorm2D, ConvLayer1D, ConvLayer2D, FFN)


class HSMSSD(nn.Module):
    """Hidden State Mixer – State Space Duality module.

    Args:
        d_model: Input feature dimension.
        ssd_expand: Internal feature expansion factor.
        A_init_range: Initialization range for the A matrix.
        state_dim: State dimension.
    """

    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16),
                 state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(
            d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(
            conv_dim, conv_dim, 3, 1, 1, groups=conv_dim,
            norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(
            d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(
            self.d_inner, d_model, 1, norm=None, act_layer=None,
            bn_weight_init=0)
        self.prev_proj = ConvLayer1D(
            self.d_inner, self.d_inner, 1, norm=None, act_layer=None,
            bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(
            *A_init_range)
        self.A = nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, H, W, h_prev=None):
        """
        Args:
            x: [B, C, L] where L = H * W.
            H, W: Spatial dimensions.
            h_prev: Previous hidden state [B, d_inner, state_dim].
        Returns:
            y: [B, C, H, W]
            h: Hidden state [B, d_inner, state_dim].
        """
        batch, _, L = x.shape
        BCdt = self.dw(
            self.BCdt_proj(x).view(batch, -1, H, W)).flatten(2)
        B, C, dt = torch.split(
            BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)

        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        AB = A * B
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(
            self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        if h_prev is not None:
            h = h + self.prev_proj(h_prev)

        y = h @ C
        y = y.view(batch, -1, H, W).contiguous()
        return y, h


class FreqHSMSSD(nn.Module):
    """Frequency-domain HSM-SSD variant with grouped projections.

    Args:
        d_model: Input feature dimension.
        ssd_expand: Internal feature expansion factor.
        A_init_range: Initialization range for the A matrix.
        state_dim: State dimension.
        groups: Number of groups for grouped convolutions.
    """

    def __init__(self, d_model=2, ssd_expand=1, A_init_range=(1, 16),
                 state_dim=64, groups=2):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(
            d_model, 3 * state_dim, 1, norm=None, act_layer=None,
            groups=groups)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(
            conv_dim, conv_dim, 3, 1, 1, groups=conv_dim,
            norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(
            d_model, 2 * self.d_inner, 1, norm=None, act_layer=None,
            groups=groups)
        self.out_proj = ConvLayer1D(
            self.d_inner, d_model, 1, norm=None, act_layer=None,
            bn_weight_init=0)
        self.prev_proj = ConvLayer1D(
            self.d_inner, self.d_inner, 1, norm=None, act_layer=None,
            bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(
            *A_init_range)
        self.A = nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, H, W, h_prev=None):
        batch, _, L = x.shape
        BCdt = self.dw(
            self.BCdt_proj(x).view(batch, -1, H, W)).flatten(2)
        B, C, dt = torch.split(
            BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)

        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        AB = A * B
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(
            self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        if h_prev is not None:
            h = h + self.prev_proj(h_prev)

        y = h @ C
        y = y.view(batch, -1, H, W).contiguous()
        return y, h
