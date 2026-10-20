#!/usr/bin/env python
"""
Run a trained E2E-SD model on a PartialSpoof split or on an external test set.

    python scripts/infer.py --checkpoint exp/e2esd/s777/best.pt --db-root $DB --split eval \
        --out-dir exp/e2esd/s777/eval
    python scripts/infer.py --checkpoint exp/e2esd/s777/best.pt --set-dir exp/sets/llamaps_a \
        --out-dir exp/e2esd/s777/llamaps_a

For PartialSpoof, give --db-root and --split. For an external test set, give --set-dir: a
directory from scripts/build_llamaps.py or scripts/build_partialedit.py with eval.lst,
vad/<utt>.vad, ref_spoof_dia.rttm and wav_dir.txt.

Output files in --out-dir:
  hyp_spoof_dia.rttm  hypothesis RTTM (cluster Ckk = attractor a_k)
  bona_scores.npz     bonafide score cos(e_t, a_0) of each 20 ms frame, for each utterance
  k_pred.tsv          columns utt, K_ref (reference K) and K_pred (K_hat, or oracle K with
                      --oracle-k)
  scores.npz          only with --save-scores N. The keys for each utterance are:
                        S/<utt>  cos(e_t, a_k) for k < N (float16)
                        p/<utt>  existence probabilities p_k for k < N
                        d/<utt>  duration (s)
                      scripts/sweep_k.py reads this file. It gives the results for different
                      values of theta and K_max, and for oracle K, without new inference.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.checkpoint import load_model  # noqa: E402
from e2esd.data import MAX_SECONDS, count_classes, load_wav, read_list, split_paths  # noqa: E402
from e2esd.metrics import k_stats  # noqa: E402
from e2esd.utils import labels_to_rttm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Run a trained E2E-SD model on a PartialSpoof split '
                                             'or on an external test set.')
    ap.add_argument('--checkpoint', required=True, help='checkpoint of scripts/train.py (best.pt)')
    ap.add_argument('--out-dir', required=True, help='output directory')
    ap.add_argument('--db-root', help='PartialSpoof database directory')
    ap.add_argument('--split', default='eval', choices=['train', 'dev', 'eval'],
                    help='PartialSpoof split')
    ap.add_argument('--set-dir',
                    help='external test set (scripts/build_llamaps.py or build_partialedit.py)')
    ap.add_argument('--oracle-k', action='store_true', help='use the reference K (at most K_max)')
    ap.add_argument('--threshold', type=float, help='existence threshold theta (default: config)')
    ap.add_argument('--k-max', type=int, help='K_max (default: config)')
    ap.add_argument('--save-scores', type=int, default=0, metavar='N',
                    help='also save cos(e_t, a_k) and p_k of the first N attractors to scores.npz')
    ap.add_argument('--limit', type=int,
                    help='use only the first N utterances, for a quick test. scripts/evaluate.py '
                         'still scores every utterance of the split, so its JI and JER are not '
                         'valid')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='torch device (default: cuda if available, otherwise cpu)')
    args = ap.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    if args.set_dir:
        with open(os.path.join(args.set_dir, 'wav_dir.txt')) as f:
            wav_dir = f.read().strip()
        utt_list = os.path.join(args.set_dir, 'eval.lst')
        vad_dir = os.path.join(args.set_dir, 'vad')
    else:
        if args.db_root is None:
            ap.error('give --db-root (PartialSpoof) or --set-dir')
        # The label files of the training config (data.vad_dir) define the reference K.
        p = split_paths(args.db_root, args.split, cfg.data.get('vad_dir', 'vad_20sil'))
        wav_dir, utt_list, vad_dir = p['wav'], p['list'], p['vad']
    ic = cfg.get('inference', {})
    theta = args.threshold if args.threshold is not None else ic.get('threshold', 0.5)
    k_max = args.k_max or ic.get('K_max', 8)
    sr, shift = cfg.data.sample_rate, cfg.data.frame_shift

    utts = read_list(utt_list)
    have = [u for u in utts if os.path.exists(os.path.join(vad_dir, u + '.vad'))]
    if not have:
        sys.exit(f'[infer] no utterance of {utt_list} has a label file in {vad_dir}')
    if len(have) < len(utts):
        # The PartialSpoof eval.lst lists two utterances that have no audio file and no label file.
        print(f'[infer] skipping {len(utts) - len(have)} listed utterances without labels')
    utts = have[:args.limit]
    print(f'[infer] {args.checkpoint}: {len(utts)} utterances, theta={theta}, K_max={k_max}, '
          f'{"oracle" if args.oracle_k else "estimated"} K')

    # Read all label files first. Thus, a bad label file stops the run before it writes outputs.
    k_ref = [count_classes(os.path.join(vad_dir, u + '.vad')) for u in utts]

    os.makedirs(args.out_dir, exist_ok=True)
    k_pred, bona, saved = [], {}, {}
    with open(os.path.join(args.out_dir, 'hyp_spoof_dia.rttm'), 'w') as rttm, \
            open(os.path.join(args.out_dir, 'k_pred.tsv'), 'w') as kf:
        kf.write('utt\tK_ref\tK_pred\n')
        for utt, K_ref in zip(tqdm(utts, ncols=100), k_ref):
            wav = load_wav(wav_dir, utt, sr)[:int(MAX_SECONDS * sr)]
            labels, K, S, p = model.diarize(wav.view(1, -1).to(args.device), theta, k_max,
                                            K_ref if args.oracle_k else None, args.save_scores)
            for line in labels_to_rttm(utt, labels.cpu().numpy(), shift, len(wav) / sr):
                rttm.write(line + '\n')
            kf.write(f'{utt}\t{K_ref}\t{K}\n')
            k_pred.append(K)
            bona[utt] = S[:, 0].float().cpu().numpy()
            if args.save_scores:
                saved[f'S/{utt}'] = S[:, :args.save_scores].cpu().numpy().astype(np.float16)
                saved[f'p/{utt}'] = p[:args.save_scores].cpu().numpy().astype(np.float32)
                saved[f'd/{utt}'] = np.float64(len(wav) / sr)

    np.savez_compressed(os.path.join(args.out_dir, 'bona_scores.npz'), **bona)
    if saved:
        np.savez_compressed(os.path.join(args.out_dir, 'scores.npz'), **saved)
    stats = k_stats(k_ref, k_pred)
    print(f'[infer] K accuracy {stats["K_acc"]:.2f}% (over {stats["K_over"]:.2f}%, '
          f'under {stats["K_under"]:.2f}%) -> {args.out_dir}')


if __name__ == '__main__':
    main()
