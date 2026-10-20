"""
Equal error rate (EER) from the bonafide score cos(e_t, a_0) of each 20 ms frame.

For PartialSpoof, this module computes the EERs in the same way as the official metric code
(github.com/nii-yamagishilab/PartialSpoof, BSD-3-Clause, Copyright 2023 Lin Zhang, NII):
  seg-EER  metric/SegmentEER.py at 20 ms. It uses all scored frames and the labels of
           `{split}_seglab_0.02.npy` (1 = bonafide, 0 = spoof). On each scored frame, these
           labels are equal to the official labels (`{split}_seglab_0.01.npy` min-pooled in
           pairs, 03multireso/model.py).
  utt-EER  metric/UtteranceEER.py with the 20 ms branch. The utterance score is the minimum
           frame score. The labels are those of the CM protocol (LA_* = bonafide, CON_* = spoof).
compute_det_curve and compute_eer are the ASVspoof reference implementation. The PartialSpoof
code also uses this implementation, through project-NN-Pytorch-scripts (sandbox/eval_asvspoof.py).
"""

import numpy as np

# ---------------------------------------------------------------------------------------------
# compute_det_curve and compute_eer are from eval_metrics.py of the ASVspoof 2019 t-DCF package
# (https://www.asvspoof.org/resources/tDCF_python_v2.zip). Unlike the rest of this repository
# (MIT), these two functions are licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License
# (http://creativecommons.org/licenses/by-nc-sa/4.0/).
# Changes: the scores are cast to float64, compute_eer is shortened and returns Python floats.
# ---------------------------------------------------------------------------------------------


def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001),
                                 all_scores[indices]))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """(EER in [0, 1], threshold). Target = bonafide (higher score = more bonafide)."""
    frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    i = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[i], far[i]))), float(thr[i])

# --------------------------------------- end of CC BY-NC-SA 4.0 part -------------------------


def partialspoof_eer(bona_scores, seglab_path):
    """
    Args:
        bona_scores: {utt: (T,) frame scores}
        seglab_path: the official `{split}_seglab_0.02.npy`
    Returns {'seg_EER', 'utt_EER'} (%).
    """
    seglab = np.load(seglab_path, allow_pickle=True).item()
    seg_t, seg_n, utt_t, utt_n = [], [], [], []
    for utt, s in bona_scores.items():
        if utt not in seglab:
            continue
        lab = np.asarray([int(x) for x in seglab[utt]])
        T = min(len(s), len(lab))
        seg_t.append(s[:T][lab[:T] == 1])
        seg_n.append(s[:T][lab[:T] == 0])
        (utt_t if utt.startswith('LA_') else utt_n).append(float(np.min(s)))
    return {'seg_EER': 100 * compute_eer(np.concatenate(seg_t), np.concatenate(seg_n))[0],
            'utt_EER': 100 * compute_eer(utt_t, utt_n)[0]}
