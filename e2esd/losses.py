"""
Assignment losses and class-to-attractor matching (paper Section 3.5).

Assignment loss (config key loss.assign_loss) on the frame-attractor cosine similarity S:
  scaled_ce : cross-entropy over w * S + b with learnable w and b (default)
  ce        : cross-entropy over S
  p2sgrad   : mean squared error between S and the one-hot target
  contrast  : GE2E contrast loss 1 - sigmoid(pos) + sigmoid(hardest negative) on w * S + b

Matching (config key loss.bona_match):
  fixed : class 0 (bonafide) goes to a_0. The search permutes only the other K-1 classes over
          a_1..a_{K-1} (default). Utterances without bonafide use the `free` matching.
  free  : the search tries all K! permutations of the K classes, bonafide included.
The search tries all permutations. Thus, it finds the same minimum cost as the Hungarian
algorithm (K <= 8 in PartialSpoof training).
"""

from functools import lru_cache
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledCELoss(nn.Module):
    """Cross-entropy over the scaled similarity w * S + b."""

    def __init__(self, w_init=10.0, b_init=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w_init))
        self.b = nn.Parameter(torch.tensor(b_init))

    def forward(self, S, targets):
        return F.cross_entropy(self.w * S + self.b, targets, ignore_index=-1)


class CELoss(nn.Module):
    """Cross-entropy over the raw cosine similarity."""

    def forward(self, S, targets):
        return F.cross_entropy(S, targets, ignore_index=-1)


class P2SGradLoss(nn.Module):
    """Mean squared error between the cosine similarities and the one-hot targets."""

    def forward(self, S, targets):
        valid = targets >= 0
        if not valid.any():
            return S.sum() * 0.0
        S, targets = S[valid], targets[valid]
        return F.mse_loss(S, F.one_hot(targets, S.shape[1]).to(S.dtype))


class ContrastLoss(nn.Module):
    """GE2E contrast loss on w * S + b: 1 - sigmoid(target) + sigmoid(hardest other)."""

    def __init__(self, w_init=10.0, b_init=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w_init))
        self.b = nn.Parameter(torch.tensor(b_init))

    def forward(self, S, targets):
        valid = targets >= 0
        if not valid.any():
            return S.sum() * 0.0
        S, targets = self.w * S[valid] + self.b, targets[valid]
        is_target = F.one_hot(targets, S.shape[1]).bool()
        pos = S[is_target]
        neg = S.masked_fill(is_target, float('-inf')).max(dim=-1).values
        return (1.0 - torch.sigmoid(pos) + torch.sigmoid(neg)).mean()


class AffineScore(nn.Module):
    """Gives w_b * cos(e_t, a_0) + b_b as the bonafide logit of the optional L_bin term."""

    def __init__(self, w_init=10.0, b_init=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w_init))
        self.b = nn.Parameter(torch.tensor(b_init))

    def forward(self, s0):
        return self.w * s0 + self.b


def build_assign_loss(loss_cfg):
    name = loss_cfg.get('assign_loss', 'scaled_ce')
    if name == 'scaled_ce':
        return ScaledCELoss(loss_cfg.get('w_init', 10.0), loss_cfg.get('b_init', -5.0))
    if name == 'contrast':
        return ContrastLoss(loss_cfg.get('w_init', 10.0), loss_cfg.get('b_init', -5.0))
    if name == 'p2sgrad':
        return P2SGradLoss()
    if name == 'ce':
        return CELoss()
    raise ValueError(f'unknown loss.assign_loss: {name}')


@lru_cache(maxsize=None)
def _permutations(n, device):
    """Returns the (n!, n) tensor of all permutations of range(n)."""
    return torch.tensor(list(permutations(range(n))), dtype=torch.long, device=device)


def _best_permutation(cost, n_fixed=0):
    """Returns the class-to-attractor assignment perm (K,) with the lowest total cost
    sum_c cost[c, perm[c]]. Classes 0..n_fixed-1 keep their own attractor."""
    K = cost.shape[0]
    perms = _permutations(K - n_fixed, cost.device) + n_fixed
    rows = torch.arange(n_fixed, K, device=cost.device)
    best = perms[cost[rows, perms].sum(dim=1).argmin()]
    return torch.cat([torch.arange(n_fixed, device=cost.device), best])


def match_targets(S, labels, n_classes, bona_fixed=None):
    """
    Returns the attractor target of each frame for the best class-to-attractor assignment.

    Args:
        S:          (B, T, K) frame-attractor cosine similarities
        labels:     (B, T) class of each frame (0..K_i-1, or -1 = no target)
        n_classes:  list of K_i
        bona_fixed: list of bool. True assigns class 0 to a_0 (bona_match = fixed).
    Returns:
        (B, T) attractor index of each frame (-1 = no target)
    """
    B, T, max_K = S.shape
    valid = labels >= 0
    onehot = F.one_hot(labels.clamp(min=0), max_K).float() * valid.unsqueeze(-1)
    cost = -torch.bmm(onehot.transpose(1, 2), S.detach())        # (B, class, attractor)

    targets = torch.full((B, T), -1, dtype=torch.long, device=S.device)
    for i, K in enumerate(n_classes):
        n_fixed = 1 if bona_fixed is not None and bona_fixed[i] else 0
        perm = _best_permutation(cost[i, :K, :K], n_fixed)
        targets[i, valid[i]] = perm[labels[i, valid[i]]]
    return targets
