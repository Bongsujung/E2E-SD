"""
Frame-level back-ends on top of the SSL features (config key model.backend).

  gmlp        : 5-layer gMLP (default)
  transformer : pre-LN Transformer encoder (back-end ablation)
  conformer   : Conformer encoder (back-end ablation)
"""

import torch.nn as nn

from .conformer import ConformerEncoder
from .gmlp import GMLPBackend


class TransformerBackend(nn.Module):
    def __init__(self, input_dim, emb_dim, n_layers, n_heads, ff_dim, dropout):
        super().__init__()
        self.output_dim = emb_dim
        self.proj = nn.Linear(input_dim, emb_dim)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                           dropout=dropout, activation='gelu', batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        return self.norm(self.encoder(self.proj(x)))


class ConformerBackend(nn.Module):
    def __init__(self, input_dim, emb_dim, n_layers, n_heads, ff_dim, conv_kernel, dropout):
        super().__init__()
        self.output_dim = emb_dim
        self.proj = nn.Linear(input_dim, emb_dim)
        self.conformer = ConformerEncoder(emb_dim, ff_dim, n_heads, n_layers, conv_kernel, dropout)

    def forward(self, x):
        return self.conformer(self.proj(x))


def build_backend(name, input_dim, emb_dim, n_layers, n_heads, ff_dim, conv_kernel, dropout):
    if name == 'gmlp':
        return GMLPBackend(input_dim=input_dim, n_layers=n_layers)
    if name == 'transformer':
        return TransformerBackend(input_dim, emb_dim, n_layers, n_heads, ff_dim, dropout)
    if name == 'conformer':
        return ConformerBackend(input_dim, emb_dim, n_layers, n_heads, ff_dim, conv_kernel, dropout)
    raise ValueError(f'unknown model.backend: {name}')
