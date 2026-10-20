import numpy as np
import pytest

from e2esd.metrics import compute_eer, evaluate, k_stats, partialspoof_eer, read_k_pred
from e2esd.utils import labels_to_rttm


def _rttm(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(''.join(f'SPEAKER {u} 1 {s:.4f} {d:.4f} <NA> <NA> {lab} <NA> <NA>\n'
                         for u, s, d, lab in rows))
    return str(p)


# u1: bonafide and C00: intersection 2.0, union 2.5 -> J = 0.2
#     A07 and C01 (cut to 2.5-3 s): J = 0.5. A10 and C02: J = 0
# u2: one spoofing method gets C00 (J = 0.5). The other spoofing method has no cluster (J = 1).
def test_toy_scores(tmp_path):
    ref = _rttm(tmp_path, 'ref.rttm', [
        # u1: bonafide 0-2 s, A07 2-3 s, (3-3.5 s not scored), A10 3.5-4 s
        ('u1', 0.0, 2.0, 'bonafide'), ('u1', 2.0, 1.0, 'A07'), ('u1', 3.5, 0.5, 'A10'),
        # u2: no bonafide
        ('u2', 0.0, 1.0, 'A16'), ('u2', 1.0, 1.0, 'A08'),
    ])
    hyp = _rttm(tmp_path, 'hyp.rttm', [
        ('u1', 0.0, 2.5, 'C00'), ('u1', 2.5, 1.0, 'C01'), ('u1', 3.5, 0.5, 'C02'),
        ('u2', 0.0, 2.0, 'C00'),                    # one cluster for both spoofing methods
    ])
    summary, per = evaluate(ref, hyp)                            # summary in %, per in [0, 1]
    assert summary['JI'] == pytest.approx(20, abs=1e-2)          # u2 has no bonafide
    assert summary['JER'] == pytest.approx(100 * (0.5 + 0.0 + 0.5 + 1.0) / 4, abs=1e-2)
    assert summary['n_bona_utts'] == 1 and summary['n_spoof_terms'] == 4
    assert per['u1']['A07'] == pytest.approx(0.5, abs=1e-4) and 'bonafide' not in per['u2']
    assert summary['JER_varied'] == pytest.approx(100 * (0.5 + per['u2']['A08']) / 2, abs=1e-2)
    assert summary['JER_unknown'] == pytest.approx(0.0, abs=1e-2)
    assert summary['JER_known'] == pytest.approx(100 * per['u2']['A16'], abs=1e-2)


def test_missing_hypothesis_is_full_error(tmp_path):
    ref = _rttm(tmp_path, 'r.rttm', [('u', 0.0, 1.0, 'bonafide'), ('u', 1.0, 1.0, 'A07')])
    hyp = _rttm(tmp_path, 'h.rttm', [('other', 0.0, 1.0, 'C00')])
    s, _ = evaluate(ref, hyp)
    assert s['JI'] == pytest.approx(100) and s['JER'] == pytest.approx(100)


def test_eer():
    assert compute_eer([2, 3, 4], [-1, 0, 1])[0] == 0.0
    assert compute_eer([0, 1], [0, 1])[0] == pytest.approx(0.5)


def test_partialspoof_eer(tmp_path):
    # official 20 ms labels: 1 = bonafide, 0 = spoof. LA_* utterances are bonafide.
    seglab = tmp_path / 'seglab.npy'
    np.save(seglab, {'LA_0001': ['1', '1', '1'], 'CON_0001': ['1', '0', '0']})
    scores = {'LA_0001': np.array([0.9, 0.7, 0.8]),
              'CON_0001': np.array([0.8, -0.5, -0.6, 0.9]),   # longer than its labels
              'CON_0002': np.array([0.0])}                   # not in the labels: skipped
    eer = partialspoof_eer(scores, str(seglab))
    assert eer == {'seg_EER': 0.0, 'utt_EER': 0.0}


def test_labels_to_rttm():
    lines = labels_to_rttm('u', np.array([0, 0, 1, 1, 1, 0]), 0.02, duration=0.11)
    assert lines == ['SPEAKER u 1 0.000 0.040 <NA> <NA> C00 <NA> <NA>',
                     'SPEAKER u 1 0.040 0.060 <NA> <NA> C01 <NA> <NA>',
                     'SPEAKER u 1 0.100 0.010 <NA> <NA> C00 <NA> <NA>']
    assert len(labels_to_rttm('u', np.array([0, 1, 1]), 0.02, duration=0.02)) == 1


def test_k_stats():
    s = k_stats([1, 3, 3, 4, 4], [1, 2, 4, 5, 4])       # (K, K_hat)
    assert (s['K_acc'], s['K_over'], s['K_under']) == pytest.approx((40, 40, 20))
    assert s['K_acc_ge4'] == pytest.approx(50)
    assert 'K_acc_ge4' not in k_stats([1, 2], [1, 2])


def test_read_k_pred(tmp_path):
    p = tmp_path / 'k_pred.tsv'
    p.write_text('utt\tK_ref\tK_pred\nu1\t3\t2\nu2\t1\t1\n')
    assert read_k_pred(str(p)) == {'u1': (3, 2), 'u2': (1, 1)}
