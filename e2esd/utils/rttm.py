"""
Frame labels -> RTTM.
"""


def labels_to_rttm(utt, labels, frame_shift, duration):
    """
    RTTM lines for frame labels (cluster k -> speaker 'Ckk'). Each sequence of equal labels
    becomes one segment. The function cuts the segments at `duration` (s).
    """
    lines = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or int(labels[i]) != int(labels[start]):
            st, et = start * frame_shift, min(i * frame_shift, duration)
            if et > st:
                lines.append(f'SPEAKER {utt} 1 {st:.3f} {et - st:.3f} <NA> <NA> '
                             f'C{int(labels[start]):02d} <NA> <NA>')
            start = i
    return lines
