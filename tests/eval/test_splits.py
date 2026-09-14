import pytest

from regulars.eval.splits import FoldSplit, assert_no_leakage, person_grouped_folds, users_of
from tests.eval.synth import GROUP_SIZE, make_dyads


def test_zero_leakage_on_60_users_and_dropped_share_reported() -> None:
    dyads = make_dyads(n_groups=10, seed=3)  # 60 users, 300 directed dyads
    assert len({d.rater for d in dyads}) == 60
    folds = person_grouped_folds(dyads, n_folds=5, seed=7)
    assert len(folds) == 5
    all_test_users: set[str] = set()
    for f in folds:
        assert_no_leakage(dyads, f)
        assert users_of(dyads, f.test_idx) <= f.test_users
        assert f.n_total == len(dyads)
        # dropped straddlers are the documented cost: ~2/k(1-1/k) = 32% at k=5
        assert 0.15 < f.dropped_share < 0.5
        assert len(f.test_idx) > 0 and len(f.train_idx) > 0
        assert not (all_test_users & f.test_users)
        all_test_users |= f.test_users
    assert len(all_test_users) == 60


def test_folds_are_seeded_and_balanced() -> None:
    dyads = make_dyads(n_groups=5, seed=0)
    a = person_grouped_folds(dyads, n_folds=5, seed=1)
    b = person_grouped_folds(dyads, n_folds=5, seed=1)
    c = person_grouped_folds(dyads, n_folds=5, seed=2)
    assert [f.test_users for f in a] == [f.test_users for f in b]
    assert [f.test_users for f in a] != [f.test_users for f in c]
    sizes = {len(f.test_users) for f in a}
    assert sizes == {GROUP_SIZE * 5 // 5}


def test_assert_no_leakage_catches_a_shared_user() -> None:
    dyads = make_dyads(n_groups=2, seed=0)
    bad = FoldSplit(
        fold=0,
        test_users=frozenset({dyads[0].rater, dyads[0].ratee}),
        train_idx=(1,),  # dyad 1 shares rater u0 with dyad 0
        test_idx=(0,),
        dropped_idx=(),
    )
    with pytest.raises(AssertionError, match="users in train and test"):
        assert_no_leakage(dyads, bad)


def test_too_few_users_rejected() -> None:
    dyads = make_dyads(n_groups=1, seed=0)
    with pytest.raises(ValueError):
        person_grouped_folds(dyads, n_folds=7)
