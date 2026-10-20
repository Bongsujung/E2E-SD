"""
Encoder-decoder attractor (EDA, Horiguchi et al., 2020), paper Section 3.4.

An LSTM encoder reads the frame embeddings in a shuffled order (time shuffle). Its final state
is the initial state of an LSTM decoder. The decoder changes a sequence of zero vectors into the
attractors a_0, a_1, ... A linear layer gives each attractor an existence probability. The
decoder is recurrent, so the first k attractors do not depend on the total number that it makes.

This code follows the EncoderDecoderAttractor of BUT EEND (BUTSpeechFIT/EEND,
eend/backend/models.py). That code is under the MIT License (Copyright 2019 Hitachi, Ltd.,
Copyright 2022 Brno University of Technology). As in BUT EEND, the time shuffle applies to
training and inference.
Changes from BUT EEND:
  * This module does the shuffle. Training uses a new order for each batch. Evaluation uses one
    fixed pseudo-random order (seed EVAL_SEED), so validation and inference are reproducible.
  * This module adds the weighted existence BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EVAL_SEED = 0


class EncoderDecoderAttractor(nn.Module):
    def __init__(self, dim, time_shuffle, weighted_bce):
        super().__init__()
        self.dim = dim
        self.time_shuffle = time_shuffle
        self.weighted_bce = weighted_bce
        self.encoder = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.counter = nn.Linear(dim, 1)

    def _generate(self, E, n):
        """E: (B, T, d) -> attractors (B, n, d)."""
        B, T, _ = E.shape
        if self.time_shuffle:
            if self.training:
                perm = torch.randperm(T, device=E.device)
            else:
                # a separate generator, so evaluation does not change the global random state
                perm = torch.randperm(T, generator=torch.Generator().manual_seed(EVAL_SEED))
            E = E[:, perm.to(E.device), :]
        _, state = self.encoder(E)
        attractors, _ = self.decoder(torch.zeros(B, n, self.dim, device=E.device), state)
        return attractors

    def forward(self, E, n_classes):
        """
        Training: makes max(K_i) + 1 attractors. The existence target of utterance i is 1 for
        a_0..a_{K_i-1} and 0 for the attractors after them.
        Returns (existence loss, attractors (B, max K_i, d)).
        """
        n = max(n_classes)
        attractors = self._generate(E, n + 1)
        target = torch.zeros(E.shape[0], n + 1, device=E.device)
        for i, k in enumerate(n_classes):
            target[i, :k] = 1.0
        logits = self.counter(attractors).squeeze(-1)
        if self.weighted_bce:
            loss = self._weighted_bce(logits, target)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss, attractors[:, :-1, :]

    @staticmethod
    def _weighted_bce(logits, target):
        """BCE with one weight for each attractor slot (existence ablation). Let share be the
        fraction of positive targets of a slot in the mini-batch. If 0 < share < 0.1, the weight
        of the slot is 1 / share, at most 50. The other slots have weight 1."""
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        weights = torch.ones_like(loss)
        for k in range(logits.shape[1]):
            share = target[:, k].mean().item()
            if 0 < share < 0.1:
                weights[:, k] = min(1.0 / (share + 1e-6), 50.0)
        return (loss * weights).mean()

    @torch.no_grad()
    def estimate(self, E, n):
        """Inference: returns n attractors (B, n, d) and their existence probabilities (B, n)."""
        attractors = self._generate(E, n)
        return attractors, torch.sigmoid(self.counter(attractors).squeeze(-1))
