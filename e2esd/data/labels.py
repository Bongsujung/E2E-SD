"""
PartialSpoof v1.3 label files (*.vad) -> frame labels.

A label file has one segment on each line: start end label_id.
The label ids (timestamps_v1.3/label2num_all) are:
  0        non-speech                       -> no target
  1        bonafide
  2..20    A01..A19
  100      concatenated part (ConP)         -> a class of its own (as in 3C), not scored
  101..    non-speech in a bonafide (101) or A01.. (102..) segment -> no target. With
           nonspeech='inherit' (non-speech ablation), it gets the class of that segment.

The classes of an utterance have the numbers 0..K-1. If bonafide occurs, it is class 0. The
other classes follow, sorted by label id from low to high. The external test sets use the
same format: 1 = bonafide, and each other positive id = one generator (LlamaPartialSpoof) or
editor (PartialEdit).
"""

import numpy as np

LABEL_NAMES = {0: 'nonspeech', 1: 'bonafide', 100: 'ConP',
               **{i + 1: f'A{i:02d}' for i in range(1, 20)}}
BONAFIDE_ID = 1
CONP_ID = 100
NONSPEECH_OFFSET = 100      # label id 100 + i: non-speech in a segment of label id i
NONSPEECH_MODES = ('ignore', 'inherit')


def read_segments(vad_path):
    """[(start, end, label_id), ...] without the segments of zero length."""
    segments = []
    with open(vad_path) as f:
        for n, line in enumerate(f, 1):
            try:
                st, et, lab = line.split()
                st, et, lab = float(st), float(et), int(lab)
            except ValueError:
                raise ValueError(f'{vad_path}:{n}: a line must be "<start> <end> <label id>", '
                                 f'got {line!r}') from None
            if et > st:
                segments.append((st, et, lab))
    return segments


def apply_nonspeech(segments, mode='ignore'):
    """
    Modes:
      ignore   Non-speech has no target (paper protocol).
      inherit  Non-speech in a segment gets the class of that segment (101 -> 1,
               101 + i -> 1 + i), as in the official 20 ms segment labels. If that class has
               no speech in the utterance, its non-speech has no target. This keeps the set
               of classes the same.
    """
    if mode == 'ignore':
        return segments
    if mode != 'inherit':
        raise ValueError(f'data.nonspeech must be one of {NONSPEECH_MODES}, got {mode}')
    speech = {lab for _, _, lab in segments if 0 < lab < CONP_ID}
    out = []
    for st, et, lab in segments:
        if lab > NONSPEECH_OFFSET:
            lab -= NONSPEECH_OFFSET
            lab = lab if lab in speech else 0
        out.append((st, et, lab))
    return out


def _class_ids(segments):
    """Label ids of the classes, sorted. Bonafide (1) has the lowest id, so it is class 0."""
    return sorted({lab for _, _, lab in segments if 0 < lab <= CONP_ID})


def _frames(st, et, frame_shift, T):
    s = max(0, int(np.floor(st / frame_shift + 1e-8)))
    e = min(T, int(np.ceil(et / frame_shift - 1e-8)))
    return s, max(e, s + 1)


def frame_labels(vad_path, frame_shift=0.02, nonspeech='ignore'):
    """
    Returns:
        labels:    (T,) int64, class index of each frame (-1 = no target)
        class_ids: list of label ids (class_ids[k] = label id of class k)
    """
    segments = apply_nonspeech(read_segments(vad_path), nonspeech)
    T = int(np.ceil(max((et for _, et, _ in segments), default=0.0) / frame_shift))
    labels = np.full(T, -1, dtype=np.int64)
    ids = _class_ids(segments)
    index = {lab: k for k, lab in enumerate(ids)}
    for st, et, lab in segments:
        if lab in index:
            s, e = _frames(st, et, frame_shift, T)
            labels[s:e] = index[lab]
    return labels, ids


def binary_labels(vad_path, frame_shift=0.02):
    """
    (T,) int64 bonafide / spoof target of each frame for L_bin. Non-speech in a segment also
    gets a target:
       1  bonafide, or non-speech in bonafide
       0  A01..A19, ConP, or non-speech in them
      -1  other non-speech (label id 0), or a frame without a segment
    T is the same as in frame_labels.
    """
    segments = read_segments(vad_path)
    T = int(np.ceil(max((et for _, et, _ in segments), default=0.0) / frame_shift))
    target = np.full(T, -1, dtype=np.int64)
    for st, et, lab in segments:
        if lab == 0:
            continue
        s, e = _frames(st, et, frame_shift, T)
        target[s:e] = 1 if lab in (BONAFIDE_ID, BONAFIDE_ID + NONSPEECH_OFFSET) else 0
    return target


def count_classes(vad_path):
    """K of an utterance: the number of classes, at least 1. data.nonspeech does not change K."""
    return max(len(_class_ids(read_segments(vad_path))), 1)
