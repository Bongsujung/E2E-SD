"""
Balanced sampling of utterances with many classes (existence ablation, config key
training.balanced_sampling). Only about 4% of the PartialSpoof training utterances have K >= 4.
"""

import numpy as np
from torch.utils.data import Sampler


class BalancedSpoofSampler(Sampler):
    """One epoch contains each utterance with K <= 3 one time. It also contains the utterances
    with K >= 4, repeated so that they are `target_ratio` of the epoch."""

    def __init__(self, dataset, target_ratio=0.2):
        low = [i for i, k in enumerate(dataset.n_classes) if k <= 3]
        high = [i for i, k in enumerate(dataset.n_classes) if k >= 4]
        if not high:
            self.indices = list(range(len(dataset)))
        else:
            n_high = int(len(low) * target_ratio / (1 - target_ratio))
            self.indices = low + (high * (n_high // len(high) + 1))[:n_high]

    def __iter__(self):
        return iter(np.random.permutation(self.indices).tolist())

    def __len__(self):
        return len(self.indices)
