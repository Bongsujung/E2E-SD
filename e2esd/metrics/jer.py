"""
Spoof diarization metrics JI and JER (Zhang et al., Interspeech 2024, Eq. 1-4, and arXiv version,
App. A.1). These metrics adapt the Jaccard error rate of dscore.

  J of class c in utterance j:  J_c = (FA_c + MISS_c) / |ref_c U hyp_pi(c)|
  pi is the Hungarian mapping of the reference classes to the hypothesis clusters. It gives the
  smallest sum of J over all classes of the utterance. A reference class without a cluster has
  J = 1. A hypothesis cluster without a class adds no error.

  JI   = mean J of the bonafide class over the utterances with bonafide in the reference
  JER  = mean J over all (utterance, spoofing method) pairs (Eq. 4, micro average)
  JER of spoofing method Ai = mean J_Ai over the utterances that contain Ai (Eq. 5)
  Known / Varied / Unknown JER = mean J pooled over the spoofing methods of the group

The scored region of an utterance is the union of its reference segments. The PartialSpoof
reference RTTM contains only the segments of bonafide and of the spoofing methods. Thus, the
scoring ignores non-speech and ConP.

dscore is the scorer. The J of each class comes from the frame conversion and the contingency
matrix of dscore at a 0.1 ms step. The dscore code is an unchanged copy in e2esd/_vendor/dscore.
The class-to-cluster mapping is the same as in dscore's jer() (scipy linear_sum_assignment on
the same matrix). An assert compares the mean J of each utterance with dscore's file JER. The
default 10 ms step of dscore cannot score PartialSpoof, because some reference segments are
shorter.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .._vendor.dscore.scorelib import metrics
from .._vendor.dscore.scorelib.score import turns_to_frames
from .._vendor.dscore.scorelib.turn import Turn

BONA = 'bonafide'
STEP = 1e-4     # dscore step (s). The scorer cannot score a reference turn shorter than this.
# Groups of the spoofing methods in the ASVspoof 2019 LA evaluation set
# (Zhang et al., 2024, arXiv version, App. A.1)
GROUPS = {
    'known': {'A16', 'A19'},
    'varied': {'A07', 'A08', 'A09', 'A17'},
    'unknown': {'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A18'},
}


def parse_rttm(lines):
    """{utt: [(onset, offset, label), ...]} from RTTM lines (an open file or any iterable)."""
    data = defaultdict(list)
    for line in lines:
        p = line.split()
        if len(p) >= 8:
            st = float(p[3])
            data[p[1]].append((st, st + float(p[4]), p[7]))
    return data


def read_rttm(path):
    """parse_rttm of an RTTM file."""
    with open(path) as f:
        return parse_rttm(f)


def _scored_regions(segments):
    """The union of the reference segments: sorted [onset, offset] pairs without overlap."""
    regions = []
    for st, et, _ in sorted(segments):
        if regions and st <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], et)
        else:
            regions.append([st, et])
    return regions


def _score_utt(utt, ref_segments, hyp_segments):
    """{class label: J} of one utterance."""
    regions = _scored_regions(ref_segments)
    ref_names = sorted({lab for _, _, lab in ref_segments})
    rl = turns_to_frames([Turn(s, e, speaker_id=lab, file_id=utt) for s, e, lab in ref_segments],
                         regions, step=STEP)
    ref_d = rl.sum(0)
    if (ref_d == 0).any():
        raise RuntimeError(f'reference class shorter than the {STEP} s step')
    sl = turns_to_frames([Turn(s, e, speaker_id=lab, file_id=utt) for s, e, lab in hyp_segments],
                         regions, step=STEP)          # (frames, 0) without hypothesis turns
    sl = sl[:, sl.sum(0) > 0]
    sys_d = sl.sum(0)
    if sys_d.size == 0:
        return dict.fromkeys(ref_names, 1.0)
    cm = metrics.contingency_matrix(rl, sl)
    js = 1 - cm / (ref_d[:, None] + sys_d[None, :] - cm)
    ri, ci = linear_sum_assignment(js)
    jers = np.ones(len(ref_names))
    jers[ri] = js[ri, ci]
    file_jer, _ = metrics.jer({utt: ref_d}, {utt: sys_d}, {utt: cm})
    assert abs(file_jer[utt] - 100 * jers.mean()) < 1e-6, utt
    return dict(zip(ref_names, map(float, jers)))


def score_per_utt(ref, hyp):
    """{utt: {class label: J}} for each utterance of ref. ref and hyp come from parse_rttm."""
    return {u: _score_utt(u, ref[u], hyp.get(u, [])) for u in sorted(ref)}


def _percent(values):
    return 100 * float(np.mean(values)) if values else float('nan')


def summarize(per_utt):
    """JI, JER, the group JERs and the JER of each spoofing method (%) from the output of
    score_per_utt."""
    ji = [s[BONA] for s in per_utt.values() if BONA in s]
    spoof = [(k, v) for s in per_utt.values() for k, v in s.items() if k != BONA]
    by_method = defaultdict(list)
    for k, v in spoof:
        by_method[k].append(v)
    return {
        'JI': _percent(ji),
        'JER': _percent([v for _, v in spoof]),
        **{f'JER_{g}': _percent([v for k, v in spoof if k in m]) for g, m in GROUPS.items()},
        'JER_per_method': {k: _percent(v) for k, v in sorted(by_method.items())},
        'n_utts': len(per_utt), 'n_bona_utts': len(ji), 'n_spoof_terms': len(spoof),
    }


def evaluate(ref_rttm, hyp_rttm):
    """Returns (summary, J of each class in each utterance) for two RTTM files."""
    per = score_per_utt(read_rttm(ref_rttm), read_rttm(hyp_rttm))
    return summarize(per), per
