"""
Training and inference passes of the full model. A small module replaces the SSL front-end, so
the tests need no download and run on the CPU.
"""

import glob
import os
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

import e2esd.models.encoder as encoder_module
from e2esd.checkpoint import _convert_state_dict, load_model
from e2esd.models import E2ESD, estimate_k
from e2esd.utils import load_config, to_plain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, 'configs', 'e2esd.yaml')
CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'configs', '**', '*.yaml'), recursive=True))


class TinyUpstream(nn.Module):
    """320 samples per frame (20 ms at 16 kHz) and 3 hidden states of 1024 dimensions."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(320, 1024)

    def forward(self, wav):
        T = wav.shape[1] // 320
        h = self.proj(wav[:, :T * 320].reshape(wav.shape[0], T, 320))
        return [h, h.tanh(), h.sin()]


@pytest.fixture(autouse=True)
def tiny_ssl(monkeypatch):
    monkeypatch.setattr(encoder_module, 'build_upstream',
                        lambda name, arch_cfg=None: TinyUpstream())


@pytest.mark.parametrize('config', CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_train_step_and_inference(config):
    torch.manual_seed(0)
    model = E2ESD(load_config(config))
    wav = torch.randn(2, 320 * 50)
    labels = torch.tensor([[0] * 20 + [1] * 15 + [-1] * 5 + [2] * 10,
                           [0] * 25 + [1] * 15 + [-1] * 10])
    bin_labels = (labels == 0).long().masked_fill(labels < 0, -1)
    loss, terms = model(wav, labels, n_classes=[3, 2], has_bona=[True, True],
                        bin_labels=bin_labels)
    assert torch.isfinite(loss) and terms['assign'] > 0
    loss.backward()

    model.eval()
    frames, K, S, p = model.diarize(wav[:1], theta=0.5, k_max=8)
    assert frames.shape == (50,) and 1 <= K <= 8 and S.shape == (50, 8) and p.shape == (8,)
    assert int(frames.max()) < K
    frames, K, _, _ = model.diarize(wav[:1], oracle_k=3)
    assert K == 3 and int(frames.max()) < 3


@pytest.mark.parametrize('p, theta, k_max, K', [
    ([0.9, 0.8, 0.2, 0.1], 0.5, 8, 2),      # first attractor with p_k < theta
    ([0.2, 0.9, 0.9], 0.5, 8, 1),           # at least one class
    ([0.9] * 10, 0.5, 8, 8),                # no p_k below theta: K_max
    ([0.9, 0.9, 0.9, 0.4], 0.5, 3, 3),      # only the first K_max attractors count
    ([0.9, 0.6, 0.6, 0.1], 0.7, 8, 1),      # a different theta
    ([0.9, 0.7, 0.1], 0.7, 8, 2),           # float32 values: p_1 = theta is not below theta
])
def test_estimate_k(p, theta, k_max, K):
    assert estimate_k(torch.tensor(p), theta, k_max) == K
    assert estimate_k(np.array(p, dtype=np.float32), theta, k_max) == K


def test_checkpoint_round_trip(tmp_path):
    """A checkpoint in the format of scripts/train.py loads with weights_only=True and without a
    warning."""
    cfg = load_config(DEFAULT)
    model = E2ESD(cfg)
    path = tmp_path / 'best.pt'
    torch.save({'model': model.state_dict(), 'config': to_plain(cfg), 'ssl_arch_cfg': {},
                'epoch': 1, 'dev_loss': 0.1}, path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        loaded, _ = load_model(str(path))
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_development_checkpoint_keys():
    """A checkpoint of the development code loads. It has the gMLP keys under
    post_nets_seg.disc_0 and an unused utterance branch post_nets_utt."""
    model = E2ESD(load_config(DEFAULT))
    old = {}
    for k, v in model.state_dict().items():
        if k.startswith('encoder.backend.'):
            k = 'encoder.backend.post_nets_seg.disc_0.' + k[len('encoder.backend.'):]
        old[k] = v
    old['encoder.backend.post_nets_utt.fc.weight'] = torch.zeros(1)
    model.load_state_dict(_convert_state_dict(old), strict=True)


def test_eval_time_shuffle():
    """In evaluation, the model shuffles the EDA input in a fixed order. Two calls give the same
    result, and the global random state does not change."""
    torch.manual_seed(0)
    model = E2ESD(load_config(DEFAULT)).eval()
    wav = torch.randn(1, 320 * 50)
    state = torch.get_rng_state()
    S1, p1 = model.scores(wav, 4)
    S2, p2 = model.scores(wav, 4)
    assert torch.equal(S1, S2) and torch.equal(p1, p2)
    assert torch.equal(state, torch.get_rng_state())
    model.eda.time_shuffle = False
    S3, _ = model.scores(wav, 4)
    assert not torch.allclose(S1, S3)
