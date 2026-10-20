import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    """
    Sets the seed of python, numpy and torch. The paper runs did not use `deterministic`. Without
    it, the cuDNN LSTM and scatter kernels are not deterministic. Thus, two runs with the same
    seed give slightly different results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
