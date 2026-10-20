"""
gMLP back-end (default).

This is the frame-level gMLP of the PartialSpoof multi-resolution code (Lin Zhang, NII,
BSD-3-Clause). It has five gMLP blocks on the SSL features (1024 dimensions) and then a linear
projection to 512 dimensions. The spatial gating unit mixes the frames along time with a
(max_frames x max_frames) matrix. Thus, an input can have at most `max_frames` frames
(2001 frames = 40 s at 20 ms).
"""

import torch.nn as nn

from ._gmlp_block import GMLPBlock


class GMLPBackend(nn.Module):
    def __init__(self, input_dim, n_layers, max_frames=2001):
        super().__init__()
        self.output_dim = input_dim // 2
        self.layers = nn.Sequential(*[GMLPBlock(input_dim, input_dim // 2, max_frames)
                                      for _ in range(n_layers)])
        self.fc = nn.Linear(input_dim, input_dim // 2, bias=False)

    def forward(self, x):
        """(B, T, input_dim) -> (B, T, input_dim // 2)"""
        return self.fc(self.layers(x.transpose(0, 1)).transpose(0, 1))
