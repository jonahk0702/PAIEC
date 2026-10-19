"""Harness correctness checks (step 01). Synthetic data only; no downloads needed.

    cd lab && python -m pytest paiec_lab/tests -q
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from paiec_lab import data, metrics, protocol, splits
from paiec_lab.models.baselines import M0, M1, M2


def toy_bench(n_subjects=3, n_items=100, seed=0, name="toy"):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for i in range(n_items):
            trials = 3 if i == 0 else 1  # item 0 has repeated trials
            for _ in range(trials):
                rows.append((f"s{s}", f"i{i:03d}", "", int(rng.random() < 0.3 + 0.2 * s)))
    r = pd.DataFrame(rows, columns=["subject_id", "item_id", "interactors", "response"])
    subjects = {f"s{s}": {k: (f"model-{s}" if k == "normalized_name" else "")
                          for k in data.SUBJECT_KEYS} for s in range(n_subjects)}
    items = {f"i{i:03d}": (f"question {i}", "") for i in range(n_items)}
    return data.BenchData(name, subjects, items, r, {"domain": "toy"})


def pairs_for(bench, **kw):
    pairs, _ = protocol.make_pairs(bench, max_pairs=None, **kw)
    protocol.acquire(pairs)
    return pairs


def test_m0_is_exactly_quarter():
    pairs = pairs_for(toy_bench())
    for row in protocol.evaluate(pairs, M0()):
        for n in protocol.BUDGETS:
            assert row[f"b{n}"] == pytest.approx(0.25, abs=1e-12)
        assert row["alc"] == pytest.approx(0.25, abs=1e-12)


def test_aggregated_brier_matches_explicit():
    p = np.array([0.2, 0.7, 0.5])
    trials = [np.array([1, 0, 1]), np.array([0]), np.array([1])]
    ybar = np.array([t.mean() for t in trials])
    k = np.array([len(t) for t in trials])
    explicit = np.mean(np.concatenate([(pi - t) ** 2 for pi, t in zip(p, trials)]))
    assert metrics.pair_brier(p, ybar, k) == pytest.approx(explicit, abs=1e-12)


def test_evidence_is_nested():
    pairs = pairs_for(toy_bench())
    for cross in (False, True):
        for i in range(len(pairs)):
            prev = []
            for n in protocol.BUDGETS:
                ev = protocol.evidence(pairs, i, n, cross)
                own_prev = [e for e in prev if e[0][0] == pairs[i].subject]
                own_now = [e for e in ev if e[0][0] == pairs[i].subject]
                assert own_now[:len(own_prev)] == own_prev
                assert len(own_now) == min(n, len(pairs[i].acquired))
                prev = ev


def test_no_evaluation_label_leaks():
    bench = toy_bench()
    pairs = pairs_for(bench)
    for i, p in enumerate(pairs):
        acquired = {a for a, _ in p.acquired}
        assert not acquired & set(p.eval_items)
        assert acquired <= set(p.acq_stream)
        own = [e for e in protocol.evidence(pairs, i, 31, True) if e[0][0] == p.subject]
        own_contents = {e[0][1]["item_content"] for e in own}
        eval_contents = {bench.items[it][0] for it in p.eval_items}
        assert not own_contents & eval_contents


def test_split_is_half_and_disjoint():
    for p in pairs_for(toy_bench()):
        assert len(p.acq_stream) == 50 and len(p.eval_items) == 50
        assert not set(p.acq_stream) & set(p.eval_items)


def test_deterministic():
    a = protocol.evaluate(pairs_for(toy_bench()), M1())
    b = protocol.evaluate(pairs_for(toy_bench()), M1())
    assert [r["alc"] for r in a] == [r["alc"] for r in b]


def test_m1_semantics():
    pairs = pairs_for(toy_bench())
    p = pairs[0]
    item = p.item(p.eval_items[0])
    assert M1().predict([p.subject, item], protocol.evidence(pairs, 0, 0)) == 0.5
    first_label = p.acquired[0][1]
    assert M1().predict([p.subject, item], protocol.evidence(pairs, 0, 1)) == first_label


def test_m1_ignores_other_subjects_labels():
    pairs = pairs_for(toy_bench())
    p = pairs[0]
    item = p.item(p.eval_items[0])
    own_only = M1().predict([p.subject, item], protocol.evidence(pairs, 0, 7, False))
    with_others = M1().predict([p.subject, item], protocol.evidence(pairs, 0, 7, True))
    assert own_only == with_others


def test_policy_force_accept_fills_budget():
    pairs, _ = protocol.make_pairs(toy_bench(), max_pairs=None)
    protocol.acquire(pairs, predictor=M0(), policy=lambda *a: False)
    for p in pairs:
        assert len(p.acquired) == 31
        assert [a for a, _ in p.acquired] == p.acq_stream[-31:]


def test_policy_context_counts():
    seen = []

    def policy(inp, pred, labeled, ctx):
        seen.append(ctx)
        return True

    pairs, _ = protocol.make_pairs(toy_bench(n_subjects=1), max_pairs=None)
    protocol.acquire(pairs, predictor=M0(), policy=policy)
    assert seen[0]["labels_remaining"] == 31 and seen[0]["items_remaining"] == 50
    assert seen[-1]["labels_acquired"] == 30


def test_repeated_trial_reveal_is_one_of_the_trials():
    for p in pairs_for(toy_bench()):
        for it, y in p.acquired:
            assert y in set(p.trials[it].tolist())


def test_small_pairs_excluded():
    pairs, excluded = protocol.make_pairs(toy_bench(n_items=79), max_pairs=None)
    assert pairs == [] and excluded == 3


def test_invalid_prediction_raises():
    class Bad:
        def predict(self, inp, labeled):
            return float("nan")

    with pytest.raises(ValueError):
        protocol.evaluate(pairs_for(toy_bench()), Bad())


def test_bootstrap_identical_models_zero_gain():
    g, lo, hi = metrics.paired_bootstrap([0.2, 0.3, 0.1], [0.2, 0.3, 0.1], ["a", "a", "b"])
    assert g == lo == hi == 0.0


def test_m2_shrinks_toward_prior():
    m = M2()
    m.a, m.b = 3.0, 3.0
    pairs = pairs_for(toy_bench())
    p = pairs[0]
    item = p.item(p.eval_items[0])
    assert m.predict([p.subject, item], []) == 0.5
    one = m.predict([p.subject, item], protocol.evidence(pairs, 0, 1))
    assert 0.5 - 1 / 7 - 1e-12 <= one <= 0.5 + 1 / 7 + 1e-12


def test_make_splits_disjoint():
    summary = {f"b{i}": {"domain": f"d{i % 4}", "n_responses": i * 10} for i in range(60)}
    s = splits.make_splits(summary)
    folds = [set(v) for v in s["folds"].values()]
    lock = set(s["lockbox"])
    assert len(lock) == 6
    assert sum(len(f) for f in folds) + len(lock) == 60
    for a in range(5):
        assert not folds[a] & lock
        for b in range(a + 1, 5):
            assert not folds[a] & folds[b]
    assert set(s["screen"]) <= set().union(*folds)
    train, test = splits.fold_benchmarks(s, 0)
    assert not set(train) & set(test) and not set(train) & lock


@pytest.mark.skipif(not os.path.exists(splits.SPLITS), reason="splits.json not frozen yet")
def test_frozen_splits_valid():
    s = splits.load()
    folds = [set(v) for v in s["folds"].values()]
    lock = set(s["lockbox"])
    for a in range(len(folds)):
        assert not folds[a] & lock
        for b in range(a + 1, len(folds)):
            assert not folds[a] & folds[b]
