"""M0, M1, M2 from SPEC §6. Every model has fit(train_benchmarks) and predict(input, labeled)."""

import importlib.util
import os

import numpy as np

from .. import data

_OFFICIAL_MEAN = os.path.join(data.LAB, "..", "empirical_mean", "model.py")


def matching_labels(inp, labeled):
    """(successes, count) of labels for the same visible subject on the same benchmark."""
    subject, item = inp
    bench = item.get("benchmark_id")
    k = n = 0
    for (s, it), y in labeled:
        if s == subject and it.get("benchmark_id") == bench:
            k += int(y)
            n += 1
    return k, n


class M0:
    name = "M0"

    def fit(self, train_benchmarks):
        return self

    def predict(self, inp, labeled):
        return 0.5


class M1:
    """The official empirical-mean baseline, loaded unchanged from empirical_mean/model.py."""

    name = "M1"

    def __init__(self):
        spec = importlib.util.spec_from_file_location("official_empirical_mean", _OFFICIAL_MEAN)
        self._mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._mod)

    def fit(self, train_benchmarks):
        return self

    def predict(self, inp, labeled):
        return self._mod.predict(inp, labeled)


class M2:
    """Beta shrinkage p = (a + k) / (a + b + n); (a, b) by moments on training pair accuracies."""

    name = "M2"

    def __init__(self, min_items=80, max_pairs=20):
        self.min_items, self.max_pairs = min_items, max_pairs
        self.a = self.b = 1.0

    def fit(self, train_benchmarks):
        ps = data.pair_stats()
        ps = ps[ps.benchmark.isin(set(train_benchmarks)) & (ps.n_items >= self.min_items)]
        # Same pair cap per benchmark as evaluation, so alignment_faking (4,278 pairs) and
        # sib200 (3,341) don't set the prior on their own.
        ps = (ps.groupby("benchmark", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), self.max_pairs), random_state=0)))
        acc = (ps.n_correct / ps.n_resp).to_numpy()
        m, v = float(acc.mean()), float(acc.var())
        strength = m * (1 - m) / v - 1 if v > 0 else 2.0
        strength = max(strength, 1e-3)
        self.a, self.b = m * strength, (1 - m) * strength
        return self

    def predict(self, inp, labeled):
        k, n = matching_labels(inp, labeled)
        return (self.a + k) / (self.a + self.b + n)


MODELS = {"M0": M0, "M1": M1, "M2": M2}
