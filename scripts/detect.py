#!/usr/bin/env python
"""
Detect spoofed utterances with the bonafide attractor and compute the EER. The input is a
trial list of scripts/build_detect_lists.py.

    python scripts/detect.py --checkpoint exp/e2esd/s777/best.pt --list exp/sets/detect/19la.tsv \
        --out exp/e2esd/s777/detect_19la.tsv

The score of an utterance is the minimum frame score cos(e_t, a_0), as in the PartialSpoof
utt-EER. The EDA decodes only a_0. The script cuts utterances longer than 39.5 s, because the
gMLP back-end accepts at most 2001 frames (40 s).

The script adds one row per utterance to --out and skips the utterances that are already in
--out. Thus, if you start the script again after an interruption, it continues where it stopped.
Output columns: utt, label (1 = bonafide), score, duration (s), cut (1 = the script cut the
utterance).
With --eer-only, the script only computes the EER of the rows in --out.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.checkpoint import load_model  # noqa: E402
from e2esd.data import MAX_SECONDS, read_audio  # noqa: E402
from e2esd.metrics import compute_eer  # noqa: E402


def read_tsv(path):
    with open(path) as f:
        return [line.rstrip('\n').split('\t') for line in f]


def eer_of(path):
    rows = read_tsv(path)
    y, s = np.array([int(r[1]) for r in rows]), np.array([float(r[2]) for r in rows])
    return 100 * compute_eer(s[y == 1], s[y == 0])[0], len(rows)


def main():
    ap = argparse.ArgumentParser(description='Detect spoofed utterances with the bonafide '
                                             'attractor and compute the EER.')
    ap.add_argument('--checkpoint', help='checkpoint of scripts/train.py (best.pt)')
    ap.add_argument('--list', help='trial list of scripts/build_detect_lists.py')
    ap.add_argument('--out', required=True, help='output TSV file. New rows go to its end.')
    ap.add_argument('--eer-only', action='store_true', help='compute only the EER of --out')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='torch device (default: cuda if available, otherwise cpu)')
    args = ap.parse_args()

    if not args.eer_only:
        if not (args.checkpoint and args.list):
            ap.error('give --checkpoint and --list, or --eer-only')
        model, cfg = load_model(args.checkpoint, args.device)
        sr = cfg.data.sample_rate
        rows = read_tsv(args.list)
        done = {r[0] for r in read_tsv(args.out)} if os.path.isfile(args.out) else set()
        todo = [r for r in rows if r[0] not in done]
        print(f'[detect] {args.list}: {len(rows)} trials, {len(todo)} to score')
        with open(args.out, 'a') as f:
            for i, (utt, path, lab) in enumerate(tqdm(todo, ncols=100)):
                wav = read_audio(path, sr)
                n = len(wav)
                wav = wav[:int(MAX_SECONDS * sr)]
                S, _ = model.scores(wav.view(1, -1).to(args.device), 1)
                f.write(f'{utt}\t{lab}\t{S[:, 0].min().item():.6f}\t{n / sr:.3f}\t'
                        f'{int(n > len(wav))}\n')
                if i % 500 == 0:
                    f.flush()
    eer, n = eer_of(args.out)
    print(f'[detect] {args.out}: EER {eer:.2f}% over {n} trials')


if __name__ == '__main__':
    main()
