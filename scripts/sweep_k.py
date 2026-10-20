#!/usr/bin/env python
"""
Score the results for different values of the existence threshold theta and K_max, and for
oracle K, without new inference. The script reads scores.npz and k_pred.tsv of
`scripts/infer.py --save-scores N`, with N >= the largest K_max.

    python scripts/sweep_k.py --infer-dir exp/e2esd/s777/eval_scores --db-root $DB \
        --threshold 0.3 0.4 0.5 0.6 0.7 --k-max 6 7 8 11

For each pair (theta, K_max), the script does these steps:
  1. It gets K_hat from p_k with the rule of the model (e2esd.models.estimate_k).
  2. It gives each frame the label argmax_{k < K_hat} cos(e_t, a_k).
  3. It scores the result with the dscore scorer.
The J values of an utterance depend only on its K_hat. Thus, the script scores each
(utterance, K) pair only once. Oracle K uses the reference K, at most N.
scores.npz keeps cos(e_t, a_k) in float16. Thus, a few frame labels can differ from those of
scripts/infer.py. On the paper runs, the metrics differ from those of scripts/evaluate.py by
less than 0.01 points.
Output file: <infer-dir>/sweep_k.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.data import split_paths  # noqa: E402
from e2esd.metrics import (k_stats, parse_rttm, read_k_pred, read_rttm,  # noqa: E402
                           score_per_utt, summarize)
from e2esd.models import estimate_k  # noqa: E402
from e2esd.utils import labels_to_rttm  # noqa: E402

KEYS = ('JI', 'JER', 'JER_known', 'JER_varied', 'JER_unknown')
SHIFT = 0.02    # frame shift (s) of the scores of scripts/infer.py


def load_scores(infer_dir):
    z = np.load(os.path.join(infer_dir, 'scores.npz'))
    utts = sorted(k[2:] for k in z.files if k.startswith('S/'))
    S = {u: z['S/' + u].astype(np.float32) for u in utts}
    P = {u: z['p/' + u] for u in utts}
    D = {u: float(z['d/' + u]) for u in utts}
    k_ref = {u: ref for u, (ref, _) in read_k_pred(os.path.join(infer_dir, 'k_pred.tsv')).items()}
    return utts, S, P, D, k_ref


def evaluate_k(K_of, S, D, k_ref, ref, cache):
    """Scores K_hat = K_of[utt]. cache keeps the J values of each scored (utterance, K) pair."""
    todo = [(u, K) for u, K in K_of.items() if (u, K) not in cache]
    hyp = parse_rttm(line for u, K in todo
                     for line in labels_to_rttm(u, np.argmax(S[u][:, :K], axis=1), SHIFT, D[u]))
    per = score_per_utt({u: ref[u] for u, _ in todo if u in ref}, hyp)
    cache.update({(u, K): per[u] for u, K in todo if u in per})
    summ = summarize({u: cache[(u, K)] for u, K in K_of.items() if (u, K) in cache})
    out = {k: summ[k] for k in KEYS}
    out.update(k_stats([k_ref[u] for u in K_of], list(K_of.values())))
    return out


def main():
    ap = argparse.ArgumentParser(description='Score the results for different values of theta '
                                             'and K_max, and for oracle K, without new inference.')
    ap.add_argument('--infer-dir', required=True,
                    help='output directory of scripts/infer.py (run with --save-scores)')
    ap.add_argument('--db-root', required=True, help='PartialSpoof database directory')
    ap.add_argument('--split', default='eval', choices=['dev', 'eval'], help='PartialSpoof split')
    ap.add_argument('--threshold', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7],
                    help='values of the existence threshold theta')
    ap.add_argument('--k-max', type=int, nargs='+', default=[6, 7, 8, 11],
                    help='values of K_max')
    args = ap.parse_args()

    ref = read_rttm(split_paths(args.db_root, args.split)['ref_rttm'])
    utts, S, P, D, k_ref = load_scores(args.infer_dir)
    n_stored = min(s.shape[1] for s in S.values())
    if max(args.k_max) > n_stored:
        ap.error(f'K_max up to {max(args.k_max)} needs --save-scores >= {max(args.k_max)} '
                 f'(stored: {n_stored})')

    cache, grid = {}, []
    for k_max in args.k_max:
        for theta in args.threshold:
            K_of = {u: estimate_k(P[u], theta, k_max) for u in utts}
            r = {'theta': theta, 'K_max': k_max, **evaluate_k(K_of, S, D, k_ref, ref, cache)}
            grid.append(r)
            print(' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}'
                           for k, v in r.items()), flush=True)
    K_or = {u: min(k_ref[u], n_stored) for u in utts}
    oracle = evaluate_k(K_or, S, D, k_ref, ref, cache)
    print('oracle K', ' '.join(f'{k}={oracle[k]:.2f}' for k in KEYS))
    with open(os.path.join(args.infer_dir, 'sweep_k.json'), 'w') as f:
        json.dump({'n_utts': len(utts), 'grid': grid, 'oracle': oracle}, f, indent=2)


if __name__ == '__main__':
    main()
