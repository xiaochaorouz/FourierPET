"""
Utility layers for EfficientViM modules.
"""
import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite


class LayerNorm2D(nn.Module):
    """LayerNorm for 2D tensors (B, C, H, W)."""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for 1D tensors (B, C, L)."""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim, out_dim,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim, groups=1):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1, groups=groups)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None,
                               bn_weight_init=0, groups=groups)

    def forward(self, x):
        return self.fc2(self.fc1(x))
