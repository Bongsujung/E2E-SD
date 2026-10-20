#!/usr/bin/env python
"""
Score the output of scripts/infer.py on a PartialSpoof split.

    python scripts/evaluate.py --infer-dir exp/e2esd/s777/eval --db-root $DB --split eval

Metrics (%):
  JI, JER   JER also per group (Known, Varied, Unknown) and per spoofing method
  K_acc     percentage of utterances with K_hat = K (also K_over, K_under)
  seg_EER   seg-EER of the bonafide score cos(e_t, a_0)
  utt_EER   utt-EER of the bonafide score cos(e_t, a_0)
Output files in --infer-dir: results.json and per_utt.json (J of each class in each utterance).
results.json also gives n_utts, n_bona_utts (utterances with bonafide speech, the JI terms) and
n_spoof_terms (the JER terms).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.data import split_paths  # noqa: E402
from e2esd.metrics import evaluate, k_stats, partialspoof_eer, read_k_pred  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Score the output of scripts/infer.py on a '
                                             'PartialSpoof split.')
    ap.add_argument('--infer-dir', required=True, help='output directory of scripts/infer.py')
    ap.add_argument('--db-root', required=True, help='PartialSpoof database directory')
    ap.add_argument('--split', default='eval', choices=['dev', 'eval'], help='PartialSpoof split')
    args = ap.parse_args()

    paths = split_paths(args.db_root, args.split)
    hyp = os.path.join(args.infer_dir, 'hyp_spoof_dia.rttm')
    summary, per_utt = evaluate(paths['ref_rttm'], hyp)
    res = {k: summary[k] for k in ('JI', 'JER', 'JER_known', 'JER_varied', 'JER_unknown',
                                   'JER_per_method')}
    ref, pred = zip(*read_k_pred(os.path.join(args.infer_dir, 'k_pred.tsv')).values())
    res.update(k_stats(ref, pred))
    scores = np.load(os.path.join(args.infer_dir, 'bona_scores.npz'))
    res.update(partialspoof_eer(dict(scores), paths['seglab']))

    res.update(n_utts=summary['n_utts'], n_bona_utts=summary['n_bona_utts'],
               n_spoof_terms=summary['n_spoof_terms'])
    with open(os.path.join(args.infer_dir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.infer_dir, 'per_utt.json'), 'w') as f:
        json.dump(per_utt, f)

    print(f'PartialSpoof {args.split}, %')
    for k in ('JI', 'JER', 'JER_known', 'JER_varied', 'JER_unknown', 'K_acc', 'seg_EER', 'utt_EER'):
        print(f'  {k:12s} {res[k]:7.2f}')


if __name__ == '__main__':
    main()
