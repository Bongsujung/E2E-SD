"""
Frame encoder (paper Section 3.3): SSL front-end -> softmax-weighted sum of its hidden states
-> dropout -> back-end -> frame embeddings E (B, T, d) at 20 ms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ssl import SSL_DIM, build_upstream
from .backends import build_backend


class FrameEncoder(nn.Module):
    def __init__(self, ssl_model, ssl_finetune, backend, backend_layers, emb_dim, n_heads, ff_dim,
                 conv_kernel, dropout, ssl_arch_cfg=None):
        """
        dropout applies to the weighted SSL features. It is also the dropout inside the
        Transformer and Conformer back-ends. ssl_arch_cfg builds the SSL model without its
        pre-trained weights (no download).
        """
        super().__init__()
        self.ssl_finetune = ssl_finetune
        self.ssl = build_upstream(ssl_model, ssl_arch_cfg)
        if not ssl_finetune:
            self.ssl.requires_grad_(False)
        # one weight for each hidden state (25 for the Large models). The 30 slots match the
        # checkpoint layout.
        self.ssl_layer_weights = nn.Parameter(torch.zeros(30))
        self.dropout = nn.Dropout(dropout)
        self.backend = build_backend(backend, SSL_DIM, emb_dim, backend_layers, n_heads=n_heads,
                                     ff_dim=ff_dim, conv_kernel=conv_kernel, dropout=dropout)
        if self.backend.output_dim != emb_dim:
            raise ValueError(f'model.emb_dim={emb_dim}, but the {backend} backend outputs '
                             f'{self.backend.output_dim} dims')

    def _weighted_sum(self, hidden_states):
        # For an odd T, the wav2vec 2.0 layer inputs have one padding frame at the end (see
        # e2esd/ssl/upstream.py). The cut to the shortest hidden state removes it.
        n = min(h.shape[1] for h in hidden_states)
        stacked = torch.stack([h[:, :n] for h in hidden_states], dim=0)
        w = F.softmax(self.ssl_layer_weights[:stacked.shape[0]], dim=0)
        return (w.view(-1, 1, 1, 1) * stacked).sum(dim=0)

    def forward(self, wav):
        """
        wav: (B, samples), zero-padded to the longest utterance -> E: (B, T, emb_dim).
        The code does not mask the padding. The SSL front-end and the back-end process it as
        audio, as in the paper runs. Only the losses ignore padded frames.
        """
        with torch.set_grad_enabled(self.ssl_finetune and torch.is_grad_enabled()):
            hidden = self.ssl(wav)
        return self.backend(self.dropout(self._weighted_sum(hidden)))
