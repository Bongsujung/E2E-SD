#!/usr/bin/env python
"""
Score the output of scripts/infer.py on an external test set (scripts/build_llamaps.py or
scripts/build_partialedit.py).

    python scripts/evaluate_external.py --infer-dir exp/e2esd/s777/llamaps_a \
        --set-dir exp/sets/llamaps_a --jer-ref exp/sets/llamaps_cf/ref_spoof_dia.rttm

The script scores each labeled frame. It does not use voice activity detection. Metrics (%):
  seg_EER   seg-EER of the bonafide score cos(e_t, a_0). A frame is spoof if one or more spoof
            segments overlap it.
  utt_EER   utt-EER (utterance score = minimum frame score), only if the set has bonafide and
            spoofed utterances. An utterance is bonafide if all its labeled frames are
            bonafide. n_bona_utts and n_spoof_utts give the numbers of these utterances.
  JER, JI   same definitions as on PartialSpoof (dscore scorer)
  K_acc     percentage of utterances with K_hat = K (also K_over, K_under)
--jer-ref gives a different reference RTTM for JER, JI and K. For example, the RTTM of
llamaps_cf limits them to the partially spoofed utterances of LlamaPartialSpoof part a.
Output files in --infer-dir:
  results.json    the metrics
  ref_clean.rttm  the reference RTTM for JER, JI and K, without turns shorter than 0.1 ms
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.data import BONAFIDE_ID, read_segments  # noqa: E402
from e2esd.metrics import STEP, compute_eer, evaluate, k_stats, read_k_pred  # noqa: E402

SHIFT = 0.02    # frame shift (s) of the scores of scripts/infer.py


def bona_spoof_frames(vad_path, T):
    """
    (T,) label of each 20 ms frame: 1 = bonafide, 0 = spoof, -1 = not labeled. A frame is spoof
    if one or more spoof segments overlap it. On PartialSpoof, the EERs use the official 20 ms
    segment labels instead (e2esd.metrics.partialspoof_eer).
    """
    cover, spoof = np.zeros(T, bool), np.zeros(T, bool)
    for st, et, lab in read_segments(vad_path):
        a, b = int(np.floor(st / SHIFT + 1e-6)), int(np.ceil(et / SHIFT - 1e-6))
        cover[a:b] = True
        if lab != BONAFIDE_ID:
            spoof[a:b] = True
    y = np.full(T, -1, dtype=np.int8)
    y[cover] = 1
    y[spoof] = 0
    return y


def eers(scores, set_dir):
    seg_t, seg_n, utt_t, utt_n = [], [], [], []
    for utt, s in scores.items():
        y = bona_spoof_frames(os.path.join(set_dir, 'vad', utt + '.vad'), len(s))
        (utt_t if (y[y >= 0] == 1).all() else utt_n).append(s.min())
        seg_t.append(s[y == 1])
        seg_n.append(s[y == 0])
    out = {'seg_EER': 100 * compute_eer(np.concatenate(seg_t), np.concatenate(seg_n))[0]}
    if utt_t and utt_n:
        out.update(utt_EER=100 * compute_eer(utt_t, utt_n)[0], n_bona_utts=len(utt_t),
                   n_spoof_utts=len(utt_n))
    return out


def clean_ref(path, out_dir):
    """
    Writes <out_dir>/ref_clean.rttm: the reference RTTM without turns shorter than the 0.1 ms
    scoring step. Returns (path of the copy, number of removed turns).
    """
    out, dropped = os.path.join(out_dir, 'ref_clean.rttm'), 0
    with open(path) as f, open(out, 'w') as g:
        for line in f:
            if float(line.split()[4]) < STEP:
                dropped += 1
            else:
                g.write(line)
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description='Score the output of scripts/infer.py on an '
                                             'external test set.')
    ap.add_argument('--infer-dir', required=True, help='output directory of scripts/infer.py')
    ap.add_argument('--set-dir', required=True,
                    help='external test set (scripts/build_llamaps.py or build_partialedit.py)')
    ap.add_argument('--jer-ref',
                    help='reference RTTM for JER, JI and K (default: <set-dir>/ref_spoof_dia.rttm)')
    args = ap.parse_args()

    res = eers(dict(np.load(os.path.join(args.infer_dir, 'bona_scores.npz'))), args.set_dir)

    ref, res['ref_turns_dropped'] = clean_ref(
        args.jer_ref or os.path.join(args.set_dir, 'ref_spoof_dia.rttm'), args.infer_dir)
    summary, per_utt = evaluate(ref, os.path.join(args.infer_dir, 'hyp_spoof_dia.rttm'))
    res.update(JER=summary['JER'], JI=summary['JI'], n_utts=summary['n_utts'])

    # K uses the utterances of the JER reference
    k_pred = read_k_pred(os.path.join(args.infer_dir, 'k_pred.tsv'))
    ref_k, pred_k = zip(*(rp for u, rp in k_pred.items() if u in per_utt))
    res.update(k_stats(ref_k, pred_k))

    with open(os.path.join(args.infer_dir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}))


if __name__ == '__main__':
    main()
