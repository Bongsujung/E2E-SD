"""
Conformer encoder (back-end ablation).

This code is adapted from DiariZen (BUT Speech@FIT, MIT License),
diarizen/models/module/conformer.py. This version does not have the relative positional
encoding.
"""

import math

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.linearQ = nn.Linear(dim, dim)
        self.linearK = nn.Linear(dim, dim)
        self.linearV = nn.Linear(dim, dim)
        self.linearO = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, T, dim) -> (B, T, dim)"""
        B, T, dim = x.shape
        q, k, v = (linear(x).reshape(B, T, self.n_heads, -1).transpose(1, 2)
                   for linear in (self.linearQ, self.linearK, self.linearV))
        scores = q @ k.transpose(-2, -1) / math.sqrt(dim // self.n_heads)
        att = self.dropout(F.softmax(scores, dim=-1))
        return self.linearO((att @ v).transpose(1, 2).reshape(B, T, dim))


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConformerMHA(nn.Module):
    """A pre-norm multi-head self-attention module with a residual connection."""

    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.ln_norm = nn.LayerNorm(dim)
        self.mha = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mha(self.ln_norm(x)))


class PositionwiseFeedForward(nn.Module):
    """A feed-forward module with a half-step residual connection (Macaron style)."""

    def __init__(self, dim, ff_dim, dropout):
        super().__init__()
        self.ln_norm = nn.LayerNorm(dim)
        self.w_1 = nn.Linear(dim, ff_dim)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ff_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout1(self.swish(self.w_1(self.ln_norm(x))))
        return x + 0.5 * self.dropout2(self.w_2(y))


class ConvolutionModule(nn.Module):
    """Pointwise conv + GLU, depthwise conv + batch norm + swish, then pointwise conv, with a
    residual connection."""

    def __init__(self, dim, kernel_size, dropout):
        super().__init__()
        assert kernel_size % 2 == 1
        self.ln_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.bn_norm = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.glu(self.pointwise_conv1(self.ln_norm(x).transpose(1, 2)))
        y = self.swish(self.bn_norm(self.depthwise_conv(y)))
        y = self.dropout(self.pointwise_conv2(y))
        return x + y.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, dim, ff_dim, n_heads, kernel_size, dropout):
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(dim, ff_dim, dropout)
        self.mha = ConformerMHA(dim, n_heads, dropout)
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.ffn2 = PositionwiseFeedForward(dim, ff_dim, dropout)
        self.ln_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln_norm(self.ffn2(self.conv(self.mha(self.ffn1(x)))))


class ConformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, n_heads, n_layers, kernel_size, dropout):
        super().__init__()
        self.conformer_layer = nn.Sequential(*[
            ConformerBlock(dim, ff_dim, n_heads, kernel_size, dropout) for _ in range(n_layers)])

    def forward(self, x):
        return self.conformer_layer(x)
