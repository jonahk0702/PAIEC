"""Lockbox, 5 benchmark folds and the screen set (SPEC §3). Frozen once in results/splits.json."""

import hashlib
import json
import os

import numpy as np

from . import data

SPLITS = os.path.join(data.LAB, "results", "splits.json")


def _stratified_order(benches, summary, seed):
    """Shuffle within domain, then interleave domains so any prefix is domain-balanced."""
    rng = np.random.default_rng(seed)
    by_dom = {}
    for b in sorted(benches):
        by_dom.setdefault(summary[b]["domain"], []).append(b)
    queues = [list(rng.permutation(v)) for _, v in sorted(by_dom.items())]
    rng.shuffle(queues)
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(str(q.pop()))
    return order


def make_splits(summary, lockbox_frac=0.10, n_folds=5, n_screen=24):
    benches = sorted(summary)
    order = _stratified_order(benches, summary, seed=1)
    n_lock = int(round(lockbox_frac * len(benches)))
    lockbox = sorted(order[:n_lock])
    dev = [b for b in benches if b not in set(lockbox)]
    # Folds: within each domain, sort by size and deal round-robin from a seeded start.
    rng = np.random.default_rng(0)
    folds = {f: [] for f in range(n_folds)}
    by_dom = {}
    for b in dev:
        by_dom.setdefault(summary[b]["domain"], []).append(b)
    start = 0
    for dom in sorted(by_dom):
        members = sorted(by_dom[dom], key=lambda b: (summary[b]["n_responses"], b))
        offset = int(rng.integers(n_folds))
        for i, b in enumerate(members):
            folds[(start + offset + i) % n_folds].append(b)
        start += len(members)
    screen = sorted(_stratified_order(dev, summary, seed=2)[:n_screen])
    out = {"lockbox": lockbox, "folds": {str(f): sorted(v) for f, v in folds.items()},
           "screen": screen, "n_usable": len(benches)}
    out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


def freeze():
    if os.path.exists(SPLITS):
        raise RuntimeError(f"{SPLITS} exists and is frozen; never regenerate it")
    out = make_splits(data.benchmark_summary())
    os.makedirs(os.path.dirname(SPLITS), exist_ok=True)
    json.dump(out, open(SPLITS, "w"), indent=1)
    return out


def load():
    s = json.load(open(SPLITS))
    body = {k: v for k, v in s.items() if k != "sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != s["sha256"]:
        raise ValueError("splits.json was edited by hand")
    return s


RELATED = os.path.join(data.LAB, "results", "related_benchmarks.json")


def related_pairs():
    """Benchmark pairs sharing items or a name family (step-02 audit)."""
    if not os.path.exists(RELATED):
        raise FileNotFoundError(f"{RELATED} missing; run the step-02 overlap audit")
    return [tuple(p) for p in json.load(open(RELATED))["related"]]


def fold_benchmarks(splits, fold, screen_only=False):
    """(train, test) benchmark lists for a fold. The lockbox is in neither.

    Training also drops any benchmark related (shared items or name family) to *any* of
    the fold's test benchmarks, so duplicated questions never straddle train and test.
    """
    fold_test = set(splits["folds"][str(fold)])
    test = sorted(fold_test & set(splits["screen"])) if screen_only else sorted(fold_test)
    blocked = {b for a, c in related_pairs() for b, other in ((a, c), (c, a))
               if other in fold_test}
    train = sorted(b for f, v in splits["folds"].items() if f != str(fold) for b in v
                   if b not in blocked)
    return train, test
