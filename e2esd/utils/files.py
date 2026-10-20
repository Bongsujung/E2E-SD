"""
Load the files that torch.save wrote.
"""

import warnings

import torch


def torch_load(path, map_location='cpu'):
    """
    torch.load with weights_only=True. Thus, the load does not run code from the file. With
    torch 2.1, this load shows a TypedStorage deprecation warning. The warning comes from torch
    itself, so this function hides it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='TypedStorage is deprecated')
        return torch.load(path, map_location=map_location, weights_only=True)
