#!/usr/bin/env python
"""
Write the trial lists of the utterance-level detection sets. Each list is <out-dir>/<set>.tsv
(19la, 21la, 21df, itw) with the columns utt, audio path and label (1 = bonafide, 0 = spoof).

    python scripts/build_detect_lists.py --out-dir exp/sets/detect \
        --asv2019-la /data/LA --asv2021-la /data/ASVspoof2021_LA_eval \
        --asv2021-df /data/ASVspoof2021_DF_eval --asv2021-keys /data/keys \
        --itw /data/release_in_the_wild

Each option gives a directory with the layout of the official release:
  --asv2019-la    LA/ of LA.zip: ASVspoof2019_LA_eval/flac/, ASVspoof2019_LA_cm_protocols/
  --asv2021-la    ASVspoof2021_LA_eval/ with flac/
  --asv2021-df    ASVspoof2021_DF_eval/ with flac/ (all parts merged)
  --asv2021-keys  keys/ of LA-keys-full.tar.gz and DF-keys-full.tar.gz:
                  {LA,DF}/CM/trial_metadata.txt
  --itw           release_in_the_wild/ with meta.csv
Each list contains every trial of the evaluation set. If you do not give the option of a set,
the script does not write its list.
"""

import argparse
import csv
import os


def read_columns(path):
    with open(path) as f:
        return [line.split() for line in f]


def write(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    print(f'{path}: {len(rows)} trials, {sum(r[2] for r in rows)} bonafide')


def main():
    ap = argparse.ArgumentParser(description='Write the trial lists of the utterance-level '
                                             'detection sets.')
    ap.add_argument('--out-dir', required=True, help='output directory')
    ap.add_argument('--asv2019-la', help='ASVspoof 2019 LA (LA/ of LA.zip)')
    ap.add_argument('--asv2021-la', help='ASVspoof2021_LA_eval/')
    ap.add_argument('--asv2021-df', help='ASVspoof2021_DF_eval/')
    ap.add_argument('--asv2021-keys', help='keys/ with {LA,DF}/CM/trial_metadata.txt')
    ap.add_argument('--itw', help='release_in_the_wild/')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.asv2019_la:
        proto = os.path.join(args.asv2019_la, 'ASVspoof2019_LA_cm_protocols',
                             'ASVspoof2019.LA.cm.eval.trl.txt')
        flac = os.path.join(args.asv2019_la, 'ASVspoof2019_LA_eval', 'flac')
        rows = [(p[1], os.path.join(flac, p[1] + '.flac'), int(p[4] == 'bonafide'))
                for p in read_columns(proto)]
        write(os.path.join(args.out_dir, '19la.tsv'), rows)

    for name, track, audio in (('21la', 'LA', args.asv2021_la), ('21df', 'DF', args.asv2021_df)):
        if not audio:
            continue
        if not args.asv2021_keys:
            ap.error(f'give --asv2021-keys for {name}')
        keys = os.path.join(args.asv2021_keys, track, 'CM', 'trial_metadata.txt')
        rows = [(p[1], os.path.join(audio, 'flac', p[1] + '.flac'), int(p[5] == 'bonafide'))
                for p in read_columns(keys)]
        write(os.path.join(args.out_dir, name + '.tsv'), rows)

    if args.itw:
        with open(os.path.join(args.itw, 'meta.csv')) as f:
            rows = [(r['file'][:-4], os.path.join(args.itw, r['file']),
                     int(r['label'] == 'bona-fide')) for r in csv.DictReader(f)]
        write(os.path.join(args.out_dir, 'itw.tsv'), rows)


if __name__ == '__main__':
    main()
