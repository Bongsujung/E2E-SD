"""
SSL front-ends that return all hidden states (paper Section 3.3).

The interface is the same as that of the s3prl upstream experts (commit 7ab62aa) of the paper
runs. It does not need s3prl or fairseq.
  * hidden_states = [input of transformer layer 0, ..., input of layer L-1, encoder output],
    that is, L + 1 tensors of shape (B, T, D). The same forward hooks as in s3prl collect them.
    wav2vec 2.0 pads its layer inputs to an even number of frames. Thus, for an odd T, its L
    layer inputs have T + 1 frames. FrameEncoder cuts them to T.
  * If the pre-trained config sets `normalize`, forward() applies a layer norm to the waveform.
  * The network is the attribute `self.model`.
The model gets no padding mask. Thus, in a zero-padded batch, the model processes the padded
samples as audio, as in the paper runs. In those runs, s3prl got rows of equal length, so its
mask was all False.

This package contains copies of the model code (see THIRD_PARTY_NOTICES.md):
  * WavLM from microsoft/unilm, through s3prl
  * wav2vec 2.0 from fairseq, as copied by s3prl 0.4.17
The weights are the converted checkpoints of s3prl. For wav2vec 2.0, feature_grad_mult and
layerdrop keep their values from the pre-trained config (1.0 / 0.0). The fairseq-based s3prl
expert of the paper runs used the same values.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.files import torch_load

# Pre-trained SSL checkpoints. fetch_ssl_checkpoint() downloads each one time to
# $E2ESD_CACHE/ssl (default ~/.cache/e2esd/ssl).
SSL_URLS = {
    'wavlm_large': 'https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt',
    # wav2vec 2.0 Large (Libri-Light + CommonVoice + Switchboard + Fisher): the 3C front-end
    'wav2vec2_large_lv_fsh_swbd_cv':
        'https://huggingface.co/s3prl/converted_ckpts/resolve/main/w2v_large_lv_fsh_swbd_cv.pt',
    # XLS-R 300M (Babu et al., 2022)
    'wav2vec2_xlsr_300m': 'https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr2_300m.pt',
}
SSL_DIM = 1024


def fetch_ssl_checkpoint(name):
    """Returns the path of the pre-trained SSL checkpoint `name`. Downloads it on first use."""
    cache = os.environ.get('E2ESD_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'e2esd'))
    path = os.path.join(cache, 'ssl', f'{name}.pt')
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'[ssl] downloading {name}: {SSL_URLS[name]}')
        torch.hub.download_url_to_file(SSL_URLS[name], path)
    return path


class _HookedUpstream(nn.Module):
    """Collects the input of every transformer layer and the encoder output."""

    def _register_hidden_hooks(self):
        for layer in self.model.encoder.layers:
            layer.register_forward_hook(
                lambda m, inp, out: self._hiddens.append(inp[0].transpose(0, 1)))
        self.model.encoder.register_forward_hook(
            lambda m, inp, out: self._hiddens.append(out[0]))

    def forward(self, wav):
        """wav: (B, samples) at 16 kHz. Returns the list of hidden states."""
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape[-1:])
        self._hiddens = []
        self.model.extract_features(wav, padding_mask=None, mask=False)
        hidden, self._hiddens = self._hiddens, []
        return hidden


class WavLMUpstream(_HookedUpstream):
    def __init__(self, arch_cfg=None):
        """arch_cfg: stored WavLM config (no download). None loads the pre-trained weights."""
        super().__init__()
        from .wavlm.WavLM import WavLM, WavLMConfig
        state = None
        if arch_cfg is None:
            ckpt = torch_load(fetch_ssl_checkpoint('wavlm_large'))
            arch_cfg, state = ckpt['cfg'], ckpt['model']
        self.arch_cfg = dict(arch_cfg)
        cfg = WavLMConfig(self.arch_cfg)
        self.model = WavLM(cfg)
        if state is not None:
            self.model.load_state_dict(state)
        self.normalize = cfg.normalize
        self._register_hidden_hooks()


class Wav2Vec2Upstream(_HookedUpstream):
    def __init__(self, name, arch_cfg=None):
        """arch_cfg: stored {'task_cfg', 'model_cfg'} (no download). None loads the weights."""
        super().__init__()
        from .wav2vec2.wav2vec2_model import AudioPretrainingConfig, Wav2Vec2Config, Wav2Vec2Model
        state = None
        if arch_cfg is None:
            ckpt = torch_load(fetch_ssl_checkpoint(name))
            arch_cfg = {'task_cfg': ckpt['task_cfg'], 'model_cfg': ckpt['model_cfg']}
            state = ckpt['model_weight']
        self.arch_cfg = {'task_cfg': dict(arch_cfg['task_cfg']),
                         'model_cfg': dict(arch_cfg['model_cfg'])}
        task_cfg = _dataclass_from(AudioPretrainingConfig, self.arch_cfg['task_cfg'])
        self.model = Wav2Vec2Model(_dataclass_from(Wav2Vec2Config, self.arch_cfg['model_cfg']))
        if state is not None:
            self.model.load_state_dict(state)
        self.normalize = task_cfg.normalize
        self._register_hidden_hooks()


def _dataclass_from(dc, cfg):
    """Makes an instance of dataclass `dc` from the keys of `cfg` that `dc` defines."""
    return dc(**{k: v for k, v in cfg.items() if k in dc.__dataclass_fields__})


def build_upstream(name, arch_cfg=None):
    """name: one of SSL_URLS. arch_cfg: stored architecture (the weights come from a checkpoint)."""
    if name == 'wavlm_large':
        return WavLMUpstream(arch_cfg)
    if name in SSL_URLS:
        return Wav2Vec2Upstream(name, arch_cfg)
    raise ValueError(f'unknown model.ssl_model: {name} (choose from {sorted(SSL_URLS)})')
