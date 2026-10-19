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


def test_pair_stats_reads_only_requested_training_tables(monkeypatch):
    seen = []

    def table(bench, name, columns=None):
        seen.append(bench)
        assert bench == "train", "A held-out table was read during fitting"
        return pd.DataFrame({"subject_id": ["s", "s"], "item_id": ["a", "b"],
                             "response": [0, 1]})

    monkeypatch.setattr(data, "_table", table)
    result = data.pair_stats(["train", "train"])
    assert seen == ["train"]
    assert result.iloc[0].n_correct == 1
    assert result.iloc[0].n_items == 2


@pytest.mark.parametrize("outcome", [0, 1])
def test_m2_extreme_training_fold_has_finite_prior(monkeypatch, outcome):
    def stats(benchmarks):
        assert benchmarks == ["train"]
        return pd.DataFrame({"benchmark": ["train"], "n_items": [100],
                             "n_resp": [100], "n_correct": [100 * outcome]})

    monkeypatch.setattr(data, "pair_stats", stats)
    model = M2().fit(["train"])
    assert model.a > 0 and model.b > 0
    assert 0 < model.predict([{}, {"benchmark_id": "b"}], []) < 1


def test_m2_rejects_empty_training_fold(monkeypatch):
    monkeypatch.setattr(data, "pair_stats", lambda benches: pd.DataFrame(
        columns=["benchmark", "n_items", "n_resp", "n_correct"]))
    with pytest.raises(ValueError, match="eligible training pair"):
        M2().fit([])


def test_structured_features_are_json():
    features = {"flags": [True, None], "counts": np.array([1, 2])}
    assert json.loads(data._features(features)) == {"flags": [True, None], "counts": [1, 2]}
    assert data._features('{"ready": true}') == '{"ready": true}'
    assert data._features(None) == ""


def test_record_rejects_misaligned_comparisons_before_writing(monkeypatch, tmp_path):
    from paiec_lab import run

    monkeypatch.setattr(run, "RESULTS", str(tmp_path))
    a = {"benchmark": "bench", "subject_id": "s1"}
    b = {"benchmark": "bench", "subject_id": "s2"}
    with pytest.raises(ValueError, match="Unaligned pairs"):
        run.record("test", "test", {"M1": [a, b], "M2": [b, a]}, "M1", 0, {}, 0)
    assert not (tmp_path / "runs.jsonl").exists()


def test_freeze_refuses_to_replace_existing_split(monkeypatch, tmp_path):
    path = tmp_path / "splits.json"
    path.write_text("original")
    monkeypatch.setattr(splits, "SPLITS", str(path))
    with pytest.raises(RuntimeError, match="frozen"):
        splits.freeze()
    assert path.read_text() == "original"


def test_corrupt_split_hash_rejected(monkeypatch, tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"lockbox": [], "sha256": "wrong"}))
    monkeypatch.setattr(splits, "SPLITS", str(path))
    with pytest.raises(ValueError, match="edited by hand"):
        splits.load()


def test_related_benchmarks_dropped_from_training(monkeypatch):
    s = {"lockbox": ["L"], "screen": ["t1"],
         "folds": {"0": ["t1", "t2"], "1": ["r1", "x1"], "2": ["r2", "x2"]}}
    monkeypatch.setattr(splits, "related_pairs", lambda: [("t2", "r1"), ("r2", "t1"), ("x1", "x2")])
    train, test = splits.fold_benchmarks(s, 0)
    assert test == ["t1", "t2"]
    assert train == ["x1", "x2"]  # r1, r2 related to fold-0 test; x1~x2 relation is irrelevant here
    train_s, test_s = splits.fold_benchmarks(s, 0, screen_only=True)
    assert test_s == ["t1"] and train_s == ["x1", "x2"]  # blocks against the whole fold, not just screen


def test_committed_related_list_loads():
    pairs = splits.related_pairs()
    assert all(len(p) == 2 for p in pairs)


def test_adaptive_cross_pair_acquisition_is_blocked():
    pairs, _ = protocol.make_pairs(toy_bench(), max_pairs=None)
    with pytest.raises(NotImplementedError):
        protocol.acquire(pairs, predictor=M0(), policy=lambda *a: True, cross_pair=True)


def test_evaluate_keeps_item_ids_aligned():
    pairs = pairs_for(toy_bench())
    row = protocol.evaluate(pairs, M0())[0]
    assert row["eval_items"] == pairs[0].eval_items
    assert len(row["preds"][0]) == len(row["eval_items"]) == len(row["ybar"])


# --- attempt persistence (step-02 review, fix 3) -------------------------------------------

import argparse

from paiec_lab import run as runmod


def _attempt_env(tmp_path, monkeypatch, fail_at=None):
    monkeypatch.setattr(runmod, "ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setattr(runmod, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(data, "DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "inventory.json").write_text('{"revision": "rev-x", "dirs": {}}')
    monkeypatch.setattr(splits, "load", lambda: {"sha256": "s" * 64})

    def fake_fold(models, fold, *rest):
        if fail_at == "fold":
            raise RuntimeError("fold exploded")
        pairs = pairs_for(toy_bench())
        rows = {m: protocol.evaluate(pairs, {"M0": M0, "M1": M1}[m]()) for m in models}
        return rows, {"train": 1, "test": 1, "excluded_small_pairs": 0}

    monkeypatch.setattr(runmod, "run_fold", fake_fold)
    if fail_at == "save":
        def bad_save(d, rows):
            raise OSError("disk full")
        monkeypatch.setattr(runmod, "save_run", bad_save)
    return argparse.Namespace(step="T", card="C", models=["M0", "M1"], comparator="M1", folds=[0],
                              screen=False, cross_pair=False, max_pairs=20, eval_cap=400)


def _manifests(tmp_path):
    return [json.load(open(p)) for p in (tmp_path / "artifacts").glob("STEP-T/*/manifest.json")]


def test_attempt_ids_unique_with_same_timestamp_and_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(runmod, "ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(runmod.time, "strftime", lambda fmt: "20261019T000000")
    monkeypatch.setattr(runmod, "git_sha", lambda: "abc1234")
    ids = {runmod.new_attempt_dir("T")[0] for _ in range(20)}
    assert len(ids) == 20


def test_successful_attempt_persists_everything(tmp_path, monkeypatch):
    a = _attempt_env(tmp_path, monkeypatch)
    run_id, d = runmod.attempt(a, ["paiec_lab/run.py", "--step", "T"])
    [m] = _manifests(tmp_path)
    assert m["status"] == "finished" and m["run_id"] == run_id
    assert m["inputs"]["inventory_sha256"] and m["command"]["argv"] == ["paiec_lab/run.py", "--step", "T"]
    assert "-m paiec_lab.run --step T" in m["command"]["rerun"] and "PAIEC_DATA=" in m["command"]["rerun"]
    assert os.path.exists(os.path.join(d, "predictions.parquet"))
    lines = [json.loads(x) for x in open(tmp_path / "results" / "runs.jsonl")]
    assert {x["run_id"] for x in lines} == {run_id} and len(lines) == 2


@pytest.mark.parametrize("where", ["save", "fold"])
def test_failed_attempt_leaves_failed_manifest_and_no_score_lines(tmp_path, monkeypatch, where):
    a = _attempt_env(tmp_path, monkeypatch, fail_at=where)
    with pytest.raises((OSError, RuntimeError)):
        runmod.attempt(a, ["paiec_lab/run.py"])
    [m] = _manifests(tmp_path)
    assert m["status"] == "failed" and ("disk full" in m["error"] or "fold exploded" in m["error"])
    assert not (tmp_path / "results" / "runs.jsonl").exists()
    if where == "save":
        assert len(m["unpersisted_lines"]) == 2
