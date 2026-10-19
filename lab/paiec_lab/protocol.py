"""Local simulator of the competition protocol (SPEC §4).

A pair is one subject on one test benchmark. Its items are split 50/50 into an
acquisition pool (the candidate stream) and an evaluation pool. Labels are revealed
only from the acquisition pool. Budgets read nested prefixes of the acquired labels.
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from . import metrics

BUDGETS = (0, 1, 3, 7, 15, 31)
MAX_LABELS = 31


def _seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@dataclass
class Pair:
    subject_id: str
    bench: object  # data.BenchData
    acq_stream: list  # acquisition item ids, in candidate order
    eval_items: list  # evaluation item ids
    trials: dict  # item_id -> np.array of 0/1 outcomes (all recorded trials)
    interactors: dict  # item_id -> string
    seed: int
    acquired: list = field(default_factory=list)  # [(item_id, label)] in acquisition order

    @property
    def subject(self):
        return self.bench.subject_dict(self.subject_id)

    def item(self, item_id):
        return self.bench.item_dict(item_id, self.interactors.get(item_id, ""))

    def entry(self, item_id, label):
        return [[self.subject, self.item(item_id)], int(label)]


def make_pairs(bench, min_items=80, max_pairs=20, eval_cap=400):
    """Scored pairs for one benchmark. Returns (pairs, n_excluded_small)."""
    r = bench.responses
    counts = r.groupby("subject_id").item_id.nunique()
    eligible = sorted(counts[counts >= min_items].index)
    excluded = int((counts < min_items).sum())
    if max_pairs is not None and len(eligible) > max_pairs:
        rng = np.random.default_rng(_seed("pairs", bench.name, 3))
        eligible = sorted(rng.choice(eligible, size=max_pairs, replace=False))
    by_subject = {s: g for s, g in r[r.subject_id.isin(set(eligible))].groupby("subject_id")}
    pairs = []
    for sid in eligible:
        g = by_subject[sid]
        trials = {i: v.to_numpy() for i, v in g.groupby("item_id").response}
        inter = {}
        for i, x in zip(g.item_id.to_numpy(), g.interactors.to_numpy()):
            if x and i not in inter:
                inter[i] = x
        items = sorted(trials)
        seed = _seed(sid, bench.name)
        perm = list(np.random.default_rng(seed).permutation(items))
        n_acq = len(items) // 2
        acq, ev = perm[:n_acq], perm[n_acq:]
        stream = list(np.random.default_rng(seed + 1).permutation(acq))
        if eval_cap is not None and len(ev) > eval_cap:
            ev = list(np.random.default_rng(seed + 2).choice(ev, size=eval_cap, replace=False))
        pairs.append(Pair(sid, bench, [str(x) for x in stream], sorted(str(x) for x in ev),
                          trials, inter, seed))
    return pairs, excluded


def _reveal(pair, item_id):
    t = pair.trials[item_id]
    if len(t) == 1:
        return int(t[0])
    k = np.random.default_rng(_seed(pair.seed, "trial", item_id)).integers(len(t))
    return int(t[k])


def evidence(pairs, i, n, cross_pair=False):
    """The `labeled` list pair i sees at budget n: its first n labels (+ others' first n)."""
    own = [pairs[i].entry(it, y) for it, y in pairs[i].acquired[:n]]
    if not cross_pair:
        return own
    others = [pairs[j].entry(it, y) for j in range(len(pairs)) if j != i
              for it, y in pairs[j].acquired[:n]]
    return own + others


def acquire(pairs, predictor=None, policy=None, mode="interleaved", cross_pair=False,
            max_labels=MAX_LABELS):
    """Fill pair.acquired for every pair. policy=None is the official random default."""
    for p in pairs:
        p.acquired = []
    if policy is None:
        for p in pairs:
            for it in p.acq_stream[:max_labels]:
                p.acquired.append((it, _reveal(p, it)))
        return
    pos = [0] * len(pairs)

    def step(i):
        p = pairs[i]
        if pos[i] >= len(p.acq_stream) or len(p.acquired) >= max_labels:
            return False
        it = p.acq_stream[pos[i]]
        pos[i] += 1
        # Everything revealed so far: this pair's labels, plus (cross-pair) other pairs'.
        labeled = [p.entry(a, y) for a, y in p.acquired]
        if cross_pair:
            labeled += [pairs[j].entry(a, y) for j in range(len(pairs)) if j != i
                        for a, y in pairs[j].acquired]
        inp = [p.subject, p.item(it)]
        pred = predictor.predict(inp, labeled) if predictor is not None else None
        remaining = max_labels - len(p.acquired)
        items_left = len(p.acq_stream) - pos[i] + 1  # includes the current candidate
        ctx = {"subject_id": p.subject_id, "benchmark_id": p.bench.bench_id,
               "labels_acquired": len(p.acquired), "labels_remaining": remaining,
               "max_labels": max_labels, "items_remaining": items_left}
        if items_left <= remaining or policy(inp, pred, labeled, ctx):
            p.acquired.append((it, _reveal(p, it)))
        return True

    if mode == "sequential":
        for i in range(len(pairs)):
            while step(i):
                pass
    else:
        active = True
        while active:
            active = False
            for i in range(len(pairs)):
                active = step(i) or active


def _check(p):
    p = float(p)
    if not math.isfinite(p) or p < 0 or p > 1:
        raise ValueError(f"invalid prediction {p!r}")
    return p


def evaluate(pairs, predictor, budgets=BUDGETS, cross_pair=False):
    """Per-pair Brier at each budget. Predicts once per distinct evaluation item."""
    out = []
    for i, p in enumerate(pairs):
        ybar = np.array([p.trials[it].mean() for it in p.eval_items])
        k = np.array([len(p.trials[it]) for it in p.eval_items], dtype=float)
        row = {"benchmark": p.bench.name, "subject_id": p.subject_id,
               "n_eval_items": len(p.eval_items), "n_eval_resp": int(k.sum()),
               "n_acquired": len(p.acquired), "preds": {}}
        for n in budgets:
            labeled = evidence(pairs, i, n, cross_pair)
            preds = np.array([_check(predictor.predict([p.subject, p.item(it)], labeled))
                              for it in p.eval_items])
            row[f"b{n}"] = metrics.pair_brier(preds, ybar, k)
            row["preds"][n] = preds
        row["alc"] = metrics.alc({n: row[f"b{n}"] for n in budgets})
        row["ybar"], row["k"] = ybar, k
        out.append(row)
    return out
