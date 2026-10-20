"""
YAML configs.

A config can start from another config with `base: <path relative to this file>`. Its own
keys replace those of the base, and the code merges nested sections. Overrides on the command
line use dotted keys, for example `--set training.seed=1234 data.db_root=/path/to/database`.
"""

import copy
import os

import yaml


class DotDict(dict):
    """A dict with attribute access to its keys, also in nested dicts."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"config has no key '{key}'") from None
        if isinstance(val, dict) and not isinstance(val, DotDict):
            val = self[key] = DotDict(val)
        return val

    def __setattr__(self, key, val):
        self[key] = val


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('base', None)
    return _merge(_read(os.path.join(os.path.dirname(path), base)), cfg) if base else cfg


def _cast(val):
    """A string from the command line -> bool / None / int / float / str."""
    low = val.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('null', 'none'):
        return None
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def parse_overrides(items):
    """['a.b=1', 'c=x'] -> {'a.b': 1, 'c': 'x'}"""
    out = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'override must be key=value, got: {item}')
        k, v = item.split('=', 1)
        out[k.strip()] = _cast(v.strip())
    return out


def load_config(path, overrides=None):
    """
    A config from a YAML file, with the overrides applied. Each override must name a key that the
    config contains. Thus, a misspelled key gives an error, and the code does not ignore it.
    """
    cfg = _read(path)
    for dotted, val in (overrides or {}).items():
        d = cfg
        *parents, last = dotted.split('.')
        for k in parents:
            if not isinstance(d.get(k), dict):
                raise KeyError(f'unknown config key: {dotted}')
            d = d[k]
        if last not in d:
            raise KeyError(f'unknown config key: {dotted}')
        d[last] = val
    return DotDict(cfg)


def to_plain(cfg):
    """The config as nested plain dicts (for json and torch.save)."""
    return {k: to_plain(v) for k, v in cfg.items()} if isinstance(cfg, dict) else cfg
