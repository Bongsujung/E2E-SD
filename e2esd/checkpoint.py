"""
Checkpoints.

scripts/train.py writes these files:
  best.pt   : {'model', 'config', 'ssl_arch_cfg', 'epoch', 'dev_loss'}       (lowest dev loss)
  latest.pt : {'model', 'config', 'ssl_arch_cfg', 'epoch', 'optimizer', 'scheduler',
               'best_dev_loss', 'epochs_without_gain'}                       (to resume training)
The checkpoint stores `ssl_arch_cfg`. Thus, load_model() builds the SSL front-end again without a
download of its pre-trained weights.
"""

from .models import E2ESD
from .utils.config import DotDict
from .utils.files import torch_load

# Old module names of the development code (for checkpoints trained before the release)
_RENAMED = (('encoder.backend.post_nets_seg.disc_0.', 'encoder.backend.'),)
_UNUSED = ('encoder.backend.post_nets_utt.',)   # utterance branch of the gMLP (not used)


def _convert_state_dict(state):
    out = {}
    for k, v in state.items():
        if k.startswith(_UNUSED):
            continue
        for old, new in _RENAMED:
            if k.startswith(old):
                k = new + k[len(old):]
        out[k] = v
    return out


def load_model(path, device='cpu'):
    """Returns (model in eval mode, config) from a checkpoint that scripts/train.py wrote."""
    ckpt = torch_load(path)
    cfg = DotDict(ckpt['config'])
    model = E2ESD(cfg, ssl_arch_cfg=ckpt.get('ssl_arch_cfg'))
    model.load_state_dict(_convert_state_dict(ckpt['model']), strict=True)
    return model.to(device).eval(), cfg
