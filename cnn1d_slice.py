"""
Stage-2 ablation: 1D CNN along the slice axis only.

The 768 ViT feature dimensions are treated as input channels and convolution
runs over the 160 slices, so, unlike cnn_mx_bigdo_ch_sw_res.py, no kernel
slides across the embedding dimension. Depth and pooling match the 2D model:
three residual conv blocks, each conv -> SiLU -> MaxPool(k=2, stride=1), with
the same slice-axis kernel sizes (10, 5, 2), followed by the same
fc1(512) -> fc2(128) -> [+sex] -> fc3 head.
"""
import torch
import torch.nn as nn

from stage2_common import RegressionHead, seed_from_env

STAGE2_SEEDABLE = True


class ConvBlock1d(nn.Module):
    """1D counterpart of ConvBlock in cnn_mx_bigdo_ch_sw_res.py."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.silu = nn.SiLU()
        self.needs_projection = (in_channels != out_channels)
        if self.needs_projection:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.projection(x) if self.needs_projection else x
        out = self.pool(self.silu(self.conv(x)))
        if identity.shape[2:] != out.shape[2:]:
            identity = nn.functional.interpolate(identity, size=out.shape[2:], mode='nearest')
        return out + identity


class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape, channels=(64, 32, 16), kernel_sizes=(10, 5, 2), dropout=0.3):
        seed_from_env()
        super().__init__()
        print("AgePredictionCNN (1D slice CNN) Shape:", input_shape)

        # input_shape: (1, num_slices, embedding_dim)
        self.num_slices = input_shape[1]
        self.embedding_dim = input_shape[2]

        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        self.conv_block1 = ConvBlock1d(self.embedding_dim, c1, kernel_size=k1)
        self.conv_block2 = ConvBlock1d(c1, c2, kernel_size=k2)
        self.conv_block3 = ConvBlock1d(c2, c3, kernel_size=k3)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            flattened_size = self._encode(torch.zeros(1, 1, self.num_slices, self.embedding_dim)).numel()
        self.head = RegressionHead(flattened_size, dropout=dropout)

    def _encode(self, x):
        # [B, 1, S, E] -> [B, E, S]: embedding dims become channels
        x = x.squeeze(1).transpose(1, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.flatten(x)

    def forward(self, x, sex):
        return self.head(self._encode(x), sex)
