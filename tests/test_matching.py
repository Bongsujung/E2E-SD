import itertools

import torch

from e2esd.losses import ScaledCELoss, match_targets


def _S(rows):
    return torch.tensor([rows], dtype=torch.float32)


def test_free_matching_ignores_class_numbering():
    S = _S([[0.9, 0.1, 0.0], [0.8, 0.2, 0.1], [0.1, 0.9, 0.2], [0.0, 0.1, 0.95]])
    base = torch.tensor([[0, 0, 1, 2]])
    ref = match_targets(S, base, [3])
    for perm in itertools.permutations(range(3)):
        relabelled = torch.tensor([[perm[c] for c in base[0].tolist()]])
        assert torch.equal(match_targets(S, relabelled, [3]), ref)


def test_fixed_assigns_bonafide_to_a0():
    # the bonafide frames are most similar to a_1, so free matching gives class 0 to a_1
    S = _S([[0.2, 0.9, 0.0], [0.1, 0.8, 0.1], [0.9, 0.1, 0.2], [0.0, 0.1, 0.95]])
    y = torch.tensor([[0, 0, 1, 2]])
    assert match_targets(S, y, [3])[0, 0].item() == 1
    fixed = match_targets(S, y, [3], bona_fixed=[True])
    assert fixed[0, :2].tolist() == [0, 0] and sorted(set(fixed[0].tolist())) == [0, 1, 2]


def test_fixed_matches_freely_without_bonafide():
    S = _S([[0.2, 0.9], [0.9, 0.1]])
    y = torch.tensor([[0, 1]])                     # class 0 is a spoofing method here
    assert torch.equal(match_targets(S, y, [2]), match_targets(S, y, [2], bona_fixed=[False]))


def test_single_class_and_frames_without_target():
    S = _S([[0.1, 0.3], [0.5, 0.2]])
    y = torch.tensor([[0, -1]])
    assert match_targets(S, y, [1], bona_fixed=[True]).tolist() == [[0, -1]]


def test_scaled_ce_ignores_frames_without_target():
    S = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    loss = ScaledCELoss()
    assert torch.allclose(loss(S, torch.tensor([0, 1, -1])), loss(S[:2], torch.tensor([0, 1])))
