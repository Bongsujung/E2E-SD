import numpy as np
import pytest

from e2esd.data.labels import binary_labels, count_classes, frame_labels, read_segments


def _vad(tmp_path, rows):
    p = tmp_path / 'u.vad'
    p.write_text(''.join(f'{s} {e} {lab}\n' for s, e, lab in rows))
    return str(p)


def test_classes_conp_and_nonspeech(tmp_path):
    # non-speech (0) | A07 (8) | ConP (100) | bonafide (1) | non-speech in A07 (108)
    p = _vad(tmp_path, [(0.0, 0.1, 0), (0.1, 0.3, 8), (0.3, 0.32, 100), (0.32, 0.6, 1),
                        (0.6, 0.7, 108)])
    y, ids = frame_labels(p, 0.02)
    assert ids == [1, 8, 100] and count_classes(p) == 3      # bonafide first, then sorted ids
    assert (y[:5] == -1).all()                               # non-speech: no target
    assert (y[5:15] == 1).all() and y[15] == 2               # A07, then ConP
    assert (y[16:30] == 0).all() and (y[30:] == -1).all()    # bonafide, then non-speech at the end


def test_nonspeech_inherit(tmp_path):
    p = _vad(tmp_path, [(0.0, 0.2, 1), (0.2, 0.3, 101), (0.3, 0.5, 8), (0.5, 0.6, 109)])
    y, ids = frame_labels(p, 0.02, nonspeech='inherit')
    assert ids == [1, 8]
    assert (y[10:15] == 0).all()                             # 101 -> bonafide
    assert (y[25:] == -1).all()                              # 109: A08 has no speech -> no target


def test_utterance_without_bonafide(tmp_path):
    p = _vad(tmp_path, [(0.0, 0.2, 9), (0.2, 0.4, 12)])
    y, ids = frame_labels(p, 0.02)
    assert ids == [9, 12] and set(np.unique(y)) == {0, 1}


def test_binary_labels(tmp_path):
    p = _vad(tmp_path, [(0.0, 0.1, 0), (0.1, 0.2, 1), (0.2, 0.3, 101), (0.3, 0.4, 100),
                        (0.4, 0.5, 8)])
    b = binary_labels(p, 0.02)
    assert (b[:5] == -1).all() and (b[5:15] == 1).all() and (b[15:] == 0).all()


def test_bad_line_names_the_file(tmp_path):
    p = tmp_path / 'u.vad'
    p.write_text('0.0 0.1 1\n0.1 0.2\n')
    with pytest.raises(ValueError, match='u.vad:2:'):
        read_segments(str(p))
