"""
Accuracy of the estimated number of classes K_hat (k_pred.tsv of scripts/infer.py).
"""

import numpy as np


def read_k_pred(path):
    """{utt: (K, K_hat)}"""
    with open(path) as f:
        rows = [line.split() for line in f][1:]
    return {r[0]: (int(r[1]), int(r[2])) for r in rows}


def k_stats(ref, pred):
    """
    Percentage of the utterances with:
      K_acc    K_hat = K
      K_over   K_hat > K
      K_under  K_hat < K
    K_acc_ge4 is K_acc of the utterances with K >= 4. The output contains K_acc_ge4 only if
    there are such utterances.
    """
    ref, pred = np.asarray(ref), np.asarray(pred)
    out = {'K_acc': 100 * float(np.mean(pred == ref)),
           'K_over': 100 * float(np.mean(pred > ref)),
           'K_under': 100 * float(np.mean(pred < ref))}
    if (ref >= 4).any():
        out['K_acc_ge4'] = 100 * float(np.mean(pred[ref >= 4] == ref[ref >= 4]))
    return out
