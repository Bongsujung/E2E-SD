import glob
import os

import pytest

from e2esd.utils import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, 'configs', 'e2esd.yaml')
CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'configs', '*', '*.yaml')))


def _flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f'{prefix}{k}.') if isinstance(v, dict) else {prefix + k: v})
    return out


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_variant_changes_one_setting(path):
    base, cfg = _flat(load_config(DEFAULT)), _flat(load_config(path))
    assert set(cfg) == set(base), 'a variant must not add keys'
    assert len([k for k in cfg if cfg[k] != base[k]]) == 1


def test_overrides():
    cfg = load_config(DEFAULT, overrides={'training.seed': 1234, 'data.db_root': '/x'})
    assert cfg.training.seed == 1234 and cfg.data.db_root == '/x'


@pytest.mark.parametrize('key', ['training.sead', 'trainin.seed', 'training.seed.x'])
def test_unknown_override_is_an_error(key):
    with pytest.raises(KeyError):
        load_config(DEFAULT, overrides={key: 1})
