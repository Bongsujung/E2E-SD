#!/usr/bin/env python
"""
Build the test sets for spoof diarization from LlamaPartialSpoof (Luong et al., ICASSP 2025).

    python scripts/build_llamaps.py --llamaps-dir /path/to/LlamaPartialSpoof --out-dir exp/sets

LlamaPartialSpoof labels only bonafide and spoof segments. The utterance id contains the
generator of the spoof segments (dev01-<generator>-partial-<splice>_<n>). Thus, the script labels
each spoof segment with its generator. A partially spoofed utterance has K = 2 (bonafide and one
generator) and no ConP class.

Sets (in <out-dir>/llamaps_<name>/):
  a   all of part a: 76,228 utterances (10,573 bonafide, 33,461 fully spoofed, 32,194 partially
      spoofed with cross-fade splices). A bonafide or fully spoofed utterance has K = 1. The
      paper uses this set for the EERs, as Luong et al. do.
  cf  the partially spoofed utterances of part a (cross-fade)
  cp  the partially spoofed cut-and-paste utterances of part b
  oa  the partially spoofed overlap-add utterances of part b
Each set contains eval.lst, vad/<utt>.vad (1 = bonafide, 2.. = generator), ref_spoof_dia.rttm
and wav_dir.txt.
"""

import argparse
import os

CLASS_ID = {'bonafide': 1, 'cosyvoice': 2, 'elevenlab': 3, 'gptsovits': 4, 'ljjets': 5,
            'xttsv2': 6, 'yourtts': 7}
SETS = {'a': 'a', 'cf': 'a', 'cp': 'b', 'oa': 'b'}     # set -> part of the release


def build(llamaps_dir, out_dir, name):
    part = SETS[name]
    d = os.path.join(out_dir, f'llamaps_{name}')
    os.makedirs(os.path.join(d, 'vad'), exist_ok=True)
    with open(os.path.join(llamaps_dir, f'label_R01TTS.0.{part}.txt')) as f:
        rows = [line.split() for line in f]
    n = 0
    with open(os.path.join(d, 'eval.lst'), 'w') as lst, \
            open(os.path.join(d, 'ref_spoof_dia.rttm'), 'w') as ref:
        for p in rows:
            utt = p[0]
            head = utt.split('_')[0]
            if name != 'a' and not head.endswith(f'-partial-{name}'):
                continue
            gen = head.split('-')[1] if head.startswith('dev01-') else None
            segs = [(float(a), float(b), 'bonafide' if lab == 'bonafide' else gen)
                    for a, b, lab in (s.split('-') for s in p[3:])]
            with open(os.path.join(d, 'vad', utt + '.vad'), 'w') as w:
                for st, et, cls in segs:
                    w.write(f'{st:.4f} {et:.4f} {CLASS_ID[cls]}\n')
                    ref.write(f'SPEAKER {utt} 1 {st:.4f} {et - st:.4f} <NA> <NA> {cls} <NA> <NA>\n')
            lst.write(utt + '\n')
            n += 1
    with open(os.path.join(d, 'wav_dir.txt'), 'w') as f:
        f.write(os.path.abspath(os.path.join(llamaps_dir, f'R01TTS.0.{part}')) + '\n')
    print(f'llamaps_{name}: {n} utterances -> {d}')


def main():
    ap = argparse.ArgumentParser(description='Build the test sets for spoof diarization from '
                                             'LlamaPartialSpoof.')
    ap.add_argument('--llamaps-dir', required=True, help='LlamaPartialSpoof release directory')
    ap.add_argument('--out-dir', required=True, help='output directory')
    ap.add_argument('--sets', nargs='+', default=list(SETS), choices=list(SETS),
                    help='sets to build (default: all)')
    args = ap.parse_args()
    for name in args.sets:
        build(args.llamaps_dir, args.out_dir, name)


if __name__ == '__main__':
    main()
