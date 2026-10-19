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


def run_fold(model_names, fold, screen, cross_pair, max_pairs, eval_cap, sp):
    train, test = splits.fold_benchmarks(sp, fold, screen_only=screen)
    models = {m: MODELS[m]().fit(train) for m in model_names}
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
    return rows, {"train": len(train), "test": len(test), "excluded_small_pairs": excluded}


def record(step, card, rows_by_model, comparator, fold_label, extra, wall):
    out_lines, sha = [], git_sha()
    os.makedirs(RESULTS, exist_ok=True)
    comp = rows_by_model.get(comparator)
    for m, rows in rows_by_model.items():
        if not rows:
            continue
        s = metrics.summarize(rows)
        line = {"step": step, "card": card, "variant": m, "fold": fold_label, "seed": 0,
                "alc": s["alc"], **{f"b{n}": s[f"b{n}"] for n in protocol.BUDGETS},
                "alc_bench": s["alc_bench"], "ece": s["ece"], "gain": None, "ci_lo": None,
                "ci_hi": None, "n_pairs": s["n_pairs"], "n_benchmarks": s["n_benchmarks"],
                "wall_s": round(wall, 1), "git_sha": sha, **extra}
        if comp is not None and m != comparator:
            g, lo, hi = metrics.paired_bootstrap([r["alc"] for r in comp], [r["alc"] for r in rows],
                                                 [r["benchmark"] for r in rows])
            line.update(gain=g, ci_lo=lo, ci_hi=hi, comparator=comparator)
        out_lines.append(line)
    with open(os.path.join(RESULTS, "runs.jsonl"), "a") as f:
        for line in out_lines:
            f.write(json.dumps(line) + "\n")
    return out_lines


def save_pairs(step, rows_by_model, tag):
    d = os.path.join(ARTIFACTS, f"STEP-{step}")
    os.makedirs(d, exist_ok=True)
    flat = [{k: v for k, v in r.items() if k not in ("preds", "ybar", "k")} | {"model": m}
            for m, rows in rows_by_model.items() for r in rows]
    path = os.path.join(d, f"pairs_{tag}.parquet")
    pd.DataFrame(flat).to_parquet(path)
    return path


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
    pooled = {m: [] for m in a.models}
    t_all = time.time()
    for f in a.folds:
        t0 = time.time()
        rows, info = run_fold(a.models, f, a.screen, a.cross_pair, a.max_pairs, a.eval_cap, sp)
        extra = {"screen": a.screen, "cross_pair": a.cross_pair, "split_sha": sp["sha256"][:12], **info}
        for line in record(a.step, a.card, rows, a.comparator, f, extra, time.time() - t0):
            print({k: line[k] for k in ("variant", "fold", "alc", "b0", "b31", "gain", "ci_lo", "ci_hi", "n_pairs")})
        for m in a.models:
            pooled[m] += rows[m]
    if len(a.folds) > 1:
        extra = {"screen": a.screen, "cross_pair": a.cross_pair, "split_sha": sp["sha256"][:12]}
        for line in record(a.step, a.card, pooled, a.comparator, "pooled", extra, time.time() - t_all):
            print({k: line[k] for k in ("variant", "fold", "alc", "b0", "b31", "gain", "ci_lo", "ci_hi", "n_pairs")})
    print(save_pairs(a.step, pooled, f"{'screen' if a.screen else 'full'}_{'x' if a.cross_pair else 'o'}"))


if __name__ == "__main__":
    main()
