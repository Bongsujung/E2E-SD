#!/usr/bin/env python
"""
Compute the mean +- sample standard deviation (n - 1) of the results over runs, as in the paper.

    python scripts/summarize.py exp/e2esd/s* --sub eval

The script reads <run>/<sub>/results.json of each run. scripts/evaluate.py and
scripts/evaluate_external.py write these files.
"""

import argparse
import json
import os

import numpy as np

ORDER = ('JI', 'JER', 'JER_known', 'JER_varied', 'JER_unknown', 'K_acc', 'K_acc_ge4', 'K_over',
         'K_under', 'seg_EER', 'utt_EER')


def read_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description='Compute the mean and standard deviation of the '
                                             'results over runs.')
    ap.add_argument('runs', nargs='+', help='run directories')
    ap.add_argument('--sub', default='eval', help='subdirectory of each run with results.json')
    ap.add_argument('--out', help='also write the summary to this JSON file')
    args = ap.parse_args()

    res = {r: read_json(os.path.join(r, args.sub, 'results.json')) for r in args.runs}
    keys = [k for k in ORDER if all(k in v for v in res.values())]
    summary = {}
    print(f'{len(res)} runs, {args.sub} (%): mean +- std')
    for k in keys:
        vals = np.array([v[k] for v in res.values()], dtype=float)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan')
        summary[k] = {'mean': float(vals.mean()), 'std': std, 'runs': vals.tolist()}
        print(f'  {k:12s} {vals.mean():7.2f} +- {std:.2f}   ' + ' '.join(f'{x:.2f}' for x in vals))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'runs': args.runs, 'sub': args.sub, 'summary': summary}, f, indent=2)


if __name__ == '__main__':
    main()
