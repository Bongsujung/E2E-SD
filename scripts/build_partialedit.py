#!/usr/bin/env python
"""
Build the test set for spoof diarization from the eval split of PartialEdit v1.1 (Zhang et al.,
Interspeech 2025, Zenodo 10.5281/zenodo.18829689).

    python scripts/build_partialedit.py --pe-dir /path/to/PartialEdit_v1.1 --out-dir exp/sets

PartialEdit replaces one or a few words of a VCTK recording with a neural speech editor
(E1 = VoiceCraft, E2 = SSR-Speech) and pastes the edited audio back. Thus, all audio outside the
edited regions is bonafide. Each utterance has one editor: K = 2, no ConP class.

Inputs in --pe-dir:
  PartialEdit_E1E2.csv  rows: <E1|E2>/<spk>/<file>.wav, the start and end of each edited
                        region, and the total duration (s)
  E1/, E2/              audio files
  spk/eval/eval.spk     speakers of the official eval split
The train and dev splits of PartialEdit use the speakers of ASVspoof 2019 and PartialSpoof.
Thus, the script uses only the eval split. Extract the files in the download directory first:

    tar xzf E1.tar.gz && tar xzf E2.tar.gz && unzip PartialEdit_spk.zip -d spk

Output files in <out-dir>/partialedit/:
  eval.lst, ref_spoof_dia.rttm, wav_dir.txt
  vad/<utt>.vad    1 = bonafide, 2 = VoiceCraft, 3 = SSR-Speech
  audio/<utt>.wav  symbolic links to the audio files, with utt = <E1|E2>_<file stem>
"""

import argparse
import csv
import os
from collections import Counter

EDITOR = {'E1': 'voicecraft', 'E2': 'ssrspeech'}
CLASS_ID = {'bonafide': 1, 'voicecraft': 2, 'ssrspeech': 3}
MIN_SEG = 0.01      # a bonafide gap of this length (s) or shorter gets no segment


def parse_row(r):
    """(editor, speaker, relative path, [(start, end), ...], total duration) of one CSV row."""
    v = [float(x) for x in r[1:]]
    if len(v) < 3 or len(v) % 2 == 0:
        raise ValueError(f'bad row length: {r}')
    total, edits = v[-1], list(zip(v[:-1:2], v[1:-1:2]))
    for i, (a, b) in enumerate(edits):
        if not 0 <= a < b <= total + 1e-6 or (i and a < edits[i - 1][1]):
            raise ValueError(f'bad edited regions: {r}')
    editor, spk = r[0].split('/')[:2]
    return editor, spk, r[0], edits, total


def segments(edits, total, name):
    """Bonafide and edited segments from 0 to total (see MIN_SEG)."""
    segs, t = [], 0.0
    for a, b in edits:
        if a - t > MIN_SEG:
            segs.append((t, a, 'bonafide'))
        segs.append((a, b, name))
        t = b
    if total - t > MIN_SEG:
        segs.append((t, total, 'bonafide'))
    return segs


def main():
    ap = argparse.ArgumentParser(description='Build the test set for spoof diarization from '
                                             'PartialEdit v1.1.')
    ap.add_argument('--pe-dir', required=True, help='extracted PartialEdit v1.1 download')
    ap.add_argument('--out-dir', required=True, help='output directory')
    args = ap.parse_args()

    with open(os.path.join(args.pe_dir, 'spk', 'eval', 'eval.spk')) as f:
        eval_spk = set(f.read().split())
    with open(os.path.join(args.pe_dir, 'PartialEdit_E1E2.csv')) as f:
        rows = list(csv.reader(f))
    d = os.path.join(args.out_dir, 'partialedit')
    audio = os.path.join(d, 'audio')
    os.makedirs(os.path.join(d, 'vad'), exist_ok=True)
    os.makedirs(audio, exist_ok=True)

    n_edits, per_editor, missing = Counter(), Counter(), []
    with open(os.path.join(d, 'eval.lst'), 'w') as lst, \
            open(os.path.join(d, 'ref_spoof_dia.rttm'), 'w') as ref:
        for r in rows:
            if not r or not r[0].endswith('.wav'):
                continue
            editor, spk, rel, edits, total = parse_row(r)
            if spk not in eval_spk:
                continue
            src = os.path.join(args.pe_dir, rel)
            if not os.path.isfile(src):
                missing.append(rel)
                continue
            utt = f'{editor}_{os.path.basename(rel)[:-4]}'
            with open(os.path.join(d, 'vad', utt + '.vad'), 'w') as w:
                for st, et, cls in segments(edits, total, EDITOR[editor]):
                    w.write(f'{st:.4f} {et:.4f} {CLASS_ID[cls]}\n')
                    ref.write(f'SPEAKER {utt} 1 {st:.4f} {et - st:.4f} <NA> <NA> {cls} <NA> <NA>\n')
            link = os.path.join(audio, utt + '.wav')
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(src), link)
            lst.write(utt + '\n')
            n_edits[len(edits)] += 1
            per_editor[editor] += 1
    with open(os.path.join(d, 'wav_dir.txt'), 'w') as f:
        f.write(os.path.abspath(audio) + '\n')
    print(f'partialedit: {sum(per_editor.values())} utterances -> {d}; per editor '
          f'{dict(per_editor)}; edited regions per utterance {dict(sorted(n_edits.items()))}')
    if missing:
        print(f'  {len(missing)} CSV rows without audio, for example {missing[:3]}')


if __name__ == '__main__':
    main()
