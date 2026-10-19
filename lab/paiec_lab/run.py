"""Run models through the simulator on held-out folds and record results (SPEC §8).

    python -m paiec_lab.run --step 01 --models M0 M1 M2 --comparator M1 --folds 0 1 --screen
    python -m paiec_lab.run --freeze-splits            # once, ever
    python -m paiec_lab.run --trace mmlu               # event trace for review

Run from lab/. All models are evaluated on the same pairs and the same acquired labels,
so gains are paired.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import data, metrics, protocol, splits
from .models.baselines import MODELS

RESULTS = os.path.join(data.LAB, "results")
ARTIFACTS = os.path.join(data.LAB, "artifacts")


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=data.LAB,
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_state():
    """Full revision plus whether tracked code differed from it when the run started."""
    try:
        full = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=data.LAB, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=data.LAB, text=True).strip()
        return {"revision": full, "dirty": bool(dirty), "dirty_files": dirty.splitlines()}
    except Exception:
        return {"revision": "unknown", "dirty": None, "dirty_files": []}


def run_fold(model_names, fold, screen, cross_pair, max_pairs, eval_cap, sp):
    train, test = splits.fold_benchmarks(sp, fold, screen_only=screen)
    models = {m: MODELS[m]().fit(train) for m in model_names}
    params = {m: (model.params() if hasattr(model, "params") else {}) for m, model in models.items()}
    rows = {m: [] for m in model_names}
    excluded = 0
    for bench in test:
        bd = data.load_benchmark(bench)
        if bd is None:
            continue
        pairs, ex = protocol.make_pairs(bd, max_pairs=max_pairs, eval_cap=eval_cap)
        excluded += ex
        if not pairs:
            continue
        protocol.acquire(pairs)  # official random default; identical for every model
        for m, model in models.items():
            rows[m] += protocol.evaluate(pairs, model, cross_pair=cross_pair)
    return rows, {"train": len(train), "test": len(test), "excluded_small_pairs": excluded,
                  "fitted": params, "train_benchmarks": train, "test_benchmarks": test}


def record(step, card, rows_by_model, comparator, fold_label, extra, wall, run_id=None):
    out_lines, sha = [], git_sha()
    os.makedirs(RESULTS, exist_ok=True)
    comp = rows_by_model.get(comparator)
    if comp is not None:
        expected = [(r["benchmark"], r["subject_id"]) for r in comp]
        if len(set(expected)) != len(expected):
            raise ValueError("Duplicate comparator pairs; check repeated folds")
        for m, rows in rows_by_model.items():
            actual = [(r["benchmark"], r["subject_id"]) for r in rows]
            if actual != expected:
                raise ValueError(f"Unaligned pairs for {m} vs {comparator}; cannot compute paired gain")
    for m, rows in rows_by_model.items():
        if not rows:
            continue
        s = metrics.summarize(rows)
        line = {"step": step, "card": card, "variant": m, "fold": fold_label, "seed": 0,
                "alc": s["alc"], **{f"b{n}": s[f"b{n}"] for n in protocol.BUDGETS},
                "alc_bench": s["alc_bench"], "ece": s["ece"], "gain": None, "ci_lo": None,
                "ci_hi": None, "n_pairs": s["n_pairs"], "n_benchmarks": s["n_benchmarks"],
                "wall_s": round(wall, 1), "git_sha": sha, "run_id": run_id, **extra}
        if comp is not None and m != comparator:
            g, lo, hi = metrics.paired_bootstrap([r["alc"] for r in comp], [r["alc"] for r in rows],
                                                 [r["benchmark"] for r in rows])
            line.update(gain=g, ci_lo=lo, ci_hi=hi, comparator=comparator)
        out_lines.append(line)
    with open(os.path.join(RESULTS, "runs.jsonl"), "a") as f:
        for line in out_lines:
            f.write(json.dumps(line) + "\n")
    return out_lines


def save_run(step, run_id, rows_by_model, manifest):
    """Per-item predictions for every model and budget, plus the run manifest. Never overwrites."""
    d = os.path.join(ARTIFACTS, f"STEP-{step}", run_id)
    os.makedirs(d, exist_ok=False)
    items = []
    for m, rows in rows_by_model.items():
        for r in rows:
            for n, preds in r["preds"].items():
                items += [{"model": m, "benchmark": r["benchmark"], "subject_id": r["subject_id"],
                           "item_id": it, "budget": n, "p": float(p), "ybar": float(y), "k": int(k)}
                          for it, p, y, k in zip(r["eval_items"], preds, r["ybar"], r["k"])]
    pd.DataFrame(items).to_parquet(os.path.join(d, "predictions.parquet"))
    pairs = [{k: v for k, v in r.items() if k not in ("preds", "ybar", "k", "eval_items")} | {"model": m}
             for m, rows in rows_by_model.items() for r in rows]
    pd.DataFrame(pairs).to_parquet(os.path.join(d, "pairs.parquet"))
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=1, default=str)
    return d


def trace(bench_name, n_subjects=2):
    """Markdown event trace: what each subject sees at each budget, cross-pair off and on."""
    bd = data.load_benchmark(bench_name)
    pairs, _ = protocol.make_pairs(bd, max_pairs=None)
    pairs = pairs[:n_subjects]
    protocol.acquire(pairs)
    lines = [f"# Event trace: {bench_name} ({bd.bench_id}), {len(pairs)} subjects\n",
             "Random default acquisition. `own` = labels of this pair; `other` = labels of other pairs.\n"]
    for p in pairs:
        lines.append(f"- subject `{p.subject_id}`: {len(p.acq_stream)} candidates, "
                     f"{len(p.eval_items)} eval items, acquired {len(p.acquired)}; "
                     f"eval ∩ acquired = {len(set(p.eval_items) & {a for a, _ in p.acquired})}")
    for cross in (False, True):
        lines.append(f"\n## cross-pair {'on' if cross else 'off'}\n")
        lines.append("| budget | subject | own labels (item:label) | other-pair labels |")
        lines.append("| --- | --- | --- | --- |")
        for n in protocol.BUDGETS:
            for i, p in enumerate(pairs):
                ev = protocol.evidence(pairs, i, n, cross)
                own = [e for e in ev if e[0][0] == p.subject]
                oth = len(ev) - len(own)
                shown = ", ".join(f"{a[:6]}:{y}" for a, y in p.acquired[:n][:5])
                more = f" … (+{n - 5})" if n > 5 else ""
                lines.append(f"| {n} | {p.subject_id[:8]} | {len(own)}: {shown}{more} | {oth} |")
    path = os.path.join(RESULTS, "STEP-01", "trace.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="01")
    ap.add_argument("--card", default="DG02")
    ap.add_argument("--models", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--comparator", default="M1")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--cross-pair", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--eval-cap", type=int, default=400)
    ap.add_argument("--freeze-splits", action="store_true")
    ap.add_argument("--trace")
    a = ap.parse_args()
    if a.freeze_splits:
        s = splits.freeze()
        print(f"frozen: lockbox {len(s['lockbox'])}, folds "
              f"{[len(v) for v in s['folds'].values()]}, screen {len(s['screen'])}")
        return
    if a.trace:
        print(trace(a.trace))
        return
    sp = splits.load()
    state = git_state()
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + git_sha()
    inv = json.load(open(os.path.join(data.DATA, "inventory.json")))
    manifest = {"run_id": run_id, "step": a.step, "card": a.card, "command": " ".join(sys.argv),
                "code": state, "data_revision": inv.get("revision"), "split_sha256": sp["sha256"],
                "related_benchmarks": splits.RELATED, "models": a.models, "comparator": a.comparator,
                "protocol": {"min_items": 80, "max_pairs": a.max_pairs, "eval_cap": a.eval_cap,
                             "acquisition": "random default", "cross_pair": a.cross_pair},
                "folds": {}, "status": "started"}
    try:
        pooled = {m: [] for m in a.models}
        t_all = time.time()
        for f in a.folds:
            t0 = time.time()
            rows, info = run_fold(a.models, f, a.screen, a.cross_pair, a.max_pairs, a.eval_cap, sp)
            manifest["folds"][str(f)] = info
            extra = {"screen": a.screen, "cross_pair": a.cross_pair, "split_sha": sp["sha256"][:12],
                     "train": info["train"], "test": info["test"],
                     "excluded_small_pairs": info["excluded_small_pairs"]}
            for line in record(a.step, a.card, rows, a.comparator, f, extra, time.time() - t0, run_id):
                print({k: line[k] for k in ("variant", "fold", "alc", "b0", "b31", "gain", "ci_lo", "ci_hi", "n_pairs")})
            for m in a.models:
                pooled[m] += rows[m]
        if len(a.folds) > 1:
            extra = {"screen": a.screen, "cross_pair": a.cross_pair, "split_sha": sp["sha256"][:12]}
            for line in record(a.step, a.card, pooled, a.comparator, "pooled", extra, time.time() - t_all,
                               run_id):
                print({k: line[k] for k in ("variant", "fold", "alc", "b0", "b31", "gain", "ci_lo", "ci_hi", "n_pairs")})
    except Exception as e:  # record failed attempts too (SPEC §8), then re-raise
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}")
        d = os.path.join(ARTIFACTS, f"STEP-{a.step}", run_id + "-failed")
        os.makedirs(d, exist_ok=True)
        json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=1, default=str)
        raise
    manifest.update(status="finished", wall_s=round(time.time() - t_all, 1))
    print(save_run(a.step, run_id, pooled, manifest))


if __name__ == "__main__":
    main()
