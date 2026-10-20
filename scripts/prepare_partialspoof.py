#!/usr/bin/env python
"""
Check the PartialSpoof layout (e2esd/data/paths.py) and write the files for scoring and for the
ConP ablation.

    python scripts/prepare_partialspoof.py --db-root /path/to/PartialSpoof/database [--noconp]

Output files in --db-root:
  * rttms/{dev,eval}/ref_spoof_dia.rttm: the reference RTTMs, with one turn per v1.3 segment of
    bonafide or A01-A19 (speaker = 'bonafide' or 'Axx'). They contain no non-speech and no
    ConP. Thus, the scoring ignores non-speech and ConP.
  * timestamps_v1.3/vad_20sil_noconp/ (only with --noconp): the v1.3 label files with ConP (100)
    changed to non-speech (0), for configs/ablation/lab_noconp.yaml.

The eval.lst file lists 71,239 utterances. But CON_E_0034982 and CON_E_0058039 have no audio
file and no label file in PartialSpoof v1.2. All scripts skip these two utterances. Thus, the
scripts score 71,237 utterances of the eval split.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.data import CONP_ID, LABEL_NAMES, read_list, read_segments, split_paths  # noqa: E402

EXPECTED_UTTS = {'train': 25380, 'dev': 24844, 'eval': 71237}


def rttm_lines(utt, vad_path):
    """RTTM lines: one turn per bonafide or A01-A19 segment. Non-speech and ConP get no turn."""
    return [f'SPEAKER {utt} 1 {st:.4f} {et - st:.4f} <NA> <NA> {LABEL_NAMES[lab]} <NA> <NA>'
            for st, et, lab in read_segments(vad_path) if 0 < lab < CONP_ID]


def build_noconp(db_root):
    src = os.path.join(db_root, 'timestamps_v1.3', 'vad_20sil')
    dst = src + '_noconp'
    for split in ('train', 'dev', 'eval'):
        os.makedirs(os.path.join(dst, split), exist_ok=True)
        files = glob.glob(os.path.join(src, split, '*.vad'))
        for path in files:
            with open(path) as f:
                out = [f'{a} {b} {0 if v == str(CONP_ID) else v}\n'
                       for a, b, v in (line.split() for line in f)]
            with open(os.path.join(dst, split, os.path.basename(path)), 'w') as f:
                f.writelines(out)
        print(f'[{split}] {len(files)} label files -> {os.path.join(dst, split)}')


def main():
    ap = argparse.ArgumentParser(description='Check the PartialSpoof layout and write the '
                                             'reference RTTMs.')
    ap.add_argument('--db-root', required=True, help='PartialSpoof database directory')
    ap.add_argument('--overwrite', action='store_true',
                    help='write the reference RTTMs again if they exist')
    ap.add_argument('--noconp', action='store_true',
                    help='also write the label files of the ConP ablation (vad_20sil_noconp)')
    args = ap.parse_args()

    ok = True
    for split in ('train', 'dev', 'eval'):
        p = split_paths(args.db_root, split)
        missing = [k for k in ('wav', 'list', 'vad', 'seglab') if not os.path.exists(p[k])]
        if missing:
            print(f'[{split}] missing: ' + ', '.join(f'{k}={p[k]}' for k in missing))
            ok = False
            continue
        utts = read_list(p['list'])
        n_wav = sum(os.path.exists(os.path.join(p['wav'], u + '.wav')) for u in utts)
        n_vad = sum(os.path.exists(os.path.join(p['vad'], u + '.vad')) for u in utts)
        good = n_wav == n_vad == EXPECTED_UTTS[split]
        ok &= good
        print(f'[{split}] listed {len(utts)}, audio {n_wav}, labels {n_vad} '
              f'(expected {EXPECTED_UTTS[split]}) {"OK" if good else "CHECK"}')
        if split == 'train':
            continue
        if os.path.exists(p['ref_rttm']) and not args.overwrite:
            print(f'[{split}] {p["ref_rttm"]} exists (--overwrite rebuilds it)')
            continue
        os.makedirs(os.path.dirname(p['ref_rttm']), exist_ok=True)
        with open(p['ref_rttm'], 'w') as f:
            for u in utts:
                vad = os.path.join(p['vad'], u + '.vad')
                if os.path.exists(vad):
                    f.writelines(line + '\n' for line in rttm_lines(u, vad))
        print(f'[{split}] wrote {p["ref_rttm"]}')
    if args.noconp:
        build_noconp(args.db_root)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
