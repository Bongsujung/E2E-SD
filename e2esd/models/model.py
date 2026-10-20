"""
E2E-SD model (paper Section 3): frame encoder + encoder-decoder attractors.

Training:  L_total = L_assign + lambda_exist * L_exist  (+ optional terms below)
Inference:
  K_hat           = min{k : p_k < theta} (at most K_max, at least 1)
  frame label     = argmax_{k < K_hat} cos(e_t, a_k)
  bonafide score  = cos(e_t, a_0)

Optional training terms (ablations):
  loss.lambda_loc > 0 : a separate linear head on E, with BCE of class 0 against the other
                        classes
  loss.lambda_bin > 0 : L_bin, BCE on w_b * cos(e_t, a_0) + b_b over every labeled frame
                        (non-speech included) of the utterances that contain bonafide
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import AffineScore, build_assign_loss, match_targets
from .eda import EncoderDecoderAttractor
from .encoder import FrameEncoder


def _fit(x, T):
    """Cuts or pads (with -1) the label tensor x (B, T') to T frames."""
    if x is None or x.shape[1] == T:
        return x
    return x[:, :T] if x.shape[1] > T else F.pad(x, (0, T - x.shape[1]), value=-1)


def estimate_k(p, theta=0.5, k_max=8):
    """
    Returns K_hat from the existence probabilities p (tensor or array). Only the first K_max
    attractors count. K_hat is the index of the first of them with p_k < theta, or K_max if none
    of them has p_k < theta. K_hat is at least 1.
    """
    below = (p[:k_max] < theta).tolist()
    return max(below.index(True), 1) if True in below else k_max


class E2ESD(nn.Module):
    def __init__(self, cfg, ssl_arch_cfg=None):
        """cfg: config with `model` and `loss` sections. ssl_arch_cfg: see FrameEncoder."""
        super().__init__()
        mc, lc = cfg.model, cfg.loss
        self.bona_match = lc.get('bona_match', 'fixed')
        if self.bona_match not in ('fixed', 'free'):
            raise ValueError(f'loss.bona_match must be fixed or free, got {self.bona_match}')
        self.lambda_exist = lc.get('lambda_exist', 1.0)
        self.lambda_loc = lc.get('lambda_loc', 0.0)
        self.lambda_bin = lc.get('lambda_bin', 0.0)
        if self.lambda_bin > 0 and self.bona_match != 'fixed':
            raise ValueError('loss.lambda_bin > 0 needs loss.bona_match = fixed')

        self.encoder = FrameEncoder(
            ssl_model=mc.ssl_model, ssl_finetune=mc.get('ssl_finetune', True),
            backend=mc.get('backend', 'gmlp'), backend_layers=mc.get('backend_layers', 5),
            emb_dim=mc.emb_dim, n_heads=mc.get('n_heads', 8), ff_dim=mc.get('ff_dim', 2048),
            conv_kernel=mc.get('conv_kernel', 31), dropout=mc.get('backend_dropout', 0.1),
            ssl_arch_cfg=ssl_arch_cfg)
        self.eda = EncoderDecoderAttractor(dim=mc.emb_dim,
                                           time_shuffle=mc.get('time_shuffle', True),
                                           weighted_bce=mc.get('weighted_exist_bce', False))
        self.assign_loss_fn = build_assign_loss(lc)
        self.loc_head = nn.Linear(mc.emb_dim, 1) if self.lambda_loc > 0 else None
        self.bin_scale = AffineScore() if self.lambda_bin > 0 else None

    @staticmethod
    def similarity(E, A):
        """Returns the cosine similarity (..., T, K) of the frame embeddings E (..., T, d) and the
        attractors A (..., K, d)."""
        return F.normalize(E, dim=-1) @ F.normalize(A, dim=-1).transpose(-2, -1)

    def forward(self, wav, labels, n_classes, has_bona, bin_labels=None):
        """
        Training loss.
            wav:        (B, samples)
            labels:     (B, T) class of each frame. Class 0 = bonafide if present, -1 = no target.
            n_classes:  list of K_i
            has_bona:   list of bool, True if the utterance contains bonafide
            bin_labels: (B, T) 1 = bonafide, 0 = spoof, -1 = unlabeled (necessary for lambda_bin)
        Returns (loss, dict of float loss terms).
        """
        E = self.encoder(wav)
        exist_loss, A = self.eda(E, n_classes)
        S = self.similarity(E, A)
        labels, bin_labels = _fit(labels, E.shape[1]), _fit(bin_labels, E.shape[1])

        bona_fixed = has_bona if self.bona_match == 'fixed' else None
        targets = match_targets(S, labels, n_classes, bona_fixed)
        # S has max_i K_i columns. For an utterance with fewer classes, the loss uses the extra
        # attractors (its stop attractor included) as negatives, as in the paper runs.
        assign_loss = self.assign_loss_fn(S.reshape(-1, S.shape[-1]), targets.reshape(-1))

        loc_loss = S.new_zeros(())
        labeled = labels >= 0
        if self.loc_head is not None and labeled.any():
            # The target is 1 for classes > 0. In utterances without bonafide (3.7% of training),
            # class 0 is a spoofing method, but it still gets target 0. The paper runs did the same.
            loc_loss = F.binary_cross_entropy_with_logits(
                self.loc_head(E).squeeze(-1)[labeled], (labels > 0).float()[labeled])

        bin_loss = S.new_zeros(())
        if self.bin_scale is not None:
            if bin_labels is None:
                raise ValueError('loss.lambda_bin > 0 needs bin_labels')
            valid = (bin_labels >= 0) & torch.tensor(has_bona, device=S.device)[:, None]
            if valid.any():
                bin_loss = F.binary_cross_entropy_with_logits(
                    self.bin_scale(S[..., 0])[valid], bin_labels[valid].float())

        total = assign_loss + self.lambda_exist * exist_loss + self.lambda_loc * loc_loss \
            + self.lambda_bin * bin_loss
        terms = {'total': total.item(), 'assign': assign_loss.item(), 'exist': exist_loss.item(),
                 'loc': loc_loss.item(), 'bin': bin_loss.item()}
        return total, terms

    @torch.no_grad()
    def scores(self, wav, n_attractors):
        """
        wav: (1, samples). Returns S (T, n) = cos(e_t, a_k) for the first n attractors and their
        existence probabilities p (n,).
        """
        E = self.encoder(wav)
        A, p = self.eda.estimate(E, n_attractors)
        return self.similarity(E[0], A[0]), p[0]

    @torch.no_grad()
    def diarize(self, wav, theta=0.5, k_max=8, oracle_k=None, n_attractors=0):
        """
        wav: (1, samples). Returns (frame labels (T,), K_hat, S (T, n), p (n,)). Label k is
        attractor a_k, and n = max(k_max, n_attractors). oracle_k replaces K_hat (at most k_max).
        """
        S, p = self.scores(wav, max(k_max, n_attractors))
        K = min(int(oracle_k), k_max) if oracle_k is not None else estimate_k(p, theta, k_max)
        return S[:, :K].argmax(dim=-1), K, S, p

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
