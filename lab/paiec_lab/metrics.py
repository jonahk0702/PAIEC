"""Exact aggregated Brier, ALC, cluster bootstrap and ECE (SPEC §5)."""

import numpy as np

WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}


def pair_brier(p, ybar, k):
    """Mean squared error over all recorded responses; an item with k trials and mean
    ybar contributes k[(p - ybar)^2 + ybar(1 - ybar)]."""
    p, ybar, k = (np.asarray(x, dtype=float) for x in (p, ybar, k))
    return float(np.sum(k * ((p - ybar) ** 2 + ybar * (1 - ybar))) / np.sum(k))


def alc(b_by_budget):
    return float(sum(WEIGHTS[n] * b_by_budget[n] for n in WEIGHTS))


def summarize(rows):
    """Headline = unweighted mean over pairs; secondary = mean over benchmarks."""
    out = {"n_pairs": len(rows), "n_benchmarks": len({r["benchmark"] for r in rows})}
    for key in ["alc"] + [f"b{n}" for n in WEIGHTS]:
        vals = np.array([r[key] for r in rows])
        out[key] = float(vals.mean())
        bench = {}
        for r in rows:
            bench.setdefault(r["benchmark"], []).append(r[key])
        out[key + "_bench"] = float(np.mean([np.mean(v) for v in bench.values()]))
    out["ece"] = ece_alc(rows)
    return out


def paired_bootstrap(comp_alc, cand_alc, benches, n=2000, seed=0):
    """GAIN = comparator - candidate (positive = candidate better). Benchmarks are the
    resampling clusters; within a resample, pairs are weighted equally."""
    gain = np.asarray(comp_alc, float) - np.asarray(cand_alc, float)
    benches = np.asarray(benches)
    uniq = np.unique(benches)
    idx = {b: np.where(benches == b)[0] for b in uniq}
    sums = np.array([gain[idx[b]].sum() for b in uniq])
    cnts = np.array([len(idx[b]) for b in uniq])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(uniq), size=(n, len(uniq)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    return float(gain.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def ece(p, ybar, k, bins=10):
    p, ybar, k = (np.asarray(x, dtype=float) for x in (p, ybar, k))
    b = np.minimum((p * bins).astype(int), bins - 1)
    total, err = k.sum(), 0.0
    for j in range(bins):
        m = b == j
        if m.any():
            w = k[m].sum()
            err += w / total * abs(np.sum(k[m] * p[m]) / w - np.sum(k[m] * ybar[m]) / w)
    return float(err)


def ece_alc(rows):
    """ECE pooled over all pairs' evaluation responses per budget, combined with ALC weights."""
    out = 0.0
    for n, w in WEIGHTS.items():
        p = np.concatenate([r["preds"][n] for r in rows])
        ybar = np.concatenate([r["ybar"] for r in rows])
        k = np.concatenate([r["k"] for r in rows])
        out += w * ece(p, ybar, k)
    return float(out)
