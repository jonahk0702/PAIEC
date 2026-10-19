"""Load measurement-db tables and build the exact platform input dicts (SPEC §2)."""

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PAIEC_DATA", os.path.join(LAB, "data"))
DB = os.path.join(DATA, "measurement-db")
CACHE = os.path.join(LAB, "artifacts", "cache")

SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra")


def _s(value):
    """Platform convention: every value is a string, missing is ""."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return ""
    return str(value)


def benchmark_hash(bench):
    return hashlib.sha1(bench.encode()).hexdigest()[:12]


def _table(bench, name, columns=None):
    path = os.path.join(DB, bench, "formatted_tables", f"{name}.parquet")
    if name == "responses" and not os.path.exists(path):
        path = os.path.join(DB, bench, "formatted_tables", "response.parquet")
    return pd.read_parquet(path, columns=columns) if os.path.exists(path) else None


def eligible_benchmarks():
    inv = json.load(open(os.path.join(DATA, "inventory.json")))
    return sorted(d for d, v in inv["dirs"].items() if v.get("eligible"))


@dataclass
class BenchData:
    """One benchmark, in memory, with platform-shaped dicts."""

    name: str
    subjects: dict  # subject_id -> subject dict (SUBJECT_KEYS, strings)
    items: dict  # item_id -> (item_content, item_features)
    responses: pd.DataFrame  # subject_id, item_id, interactors, response (0/1)
    meta: dict = field(default_factory=dict)

    @property
    def bench_id(self):
        return benchmark_hash(self.name)

    def subject_dict(self, subject_id):
        return self.subjects[subject_id]

    def item_dict(self, item_id, interactors=""):
        content, features = self.items.get(item_id, ("", ""))
        return {"item_content": content, "item_features": features,
                "interactors": interactors, "benchmark_id": self.bench_id}


def load_benchmark(bench):
    """Return BenchData, or None if the benchmark has no binary responses."""
    r = _table(bench, "responses", ["subject_id", "item_id", "interactors", "response"])
    if r is None:
        return None
    r = r.dropna(subset=["response"])
    binary = r.response.isin([0.0, 1.0])
    meta = {"dropped_non_binary": int((~binary).sum())}
    r = r[binary]
    if r.empty:
        return None
    r = pd.DataFrame({"subject_id": r.subject_id.astype(str).to_numpy(),
                      "item_id": r.item_id.astype(str).to_numpy(),
                      "interactors": [_s(v) for v in r.interactors.to_numpy()],
                      "response": r.response.astype(np.int8).to_numpy()})
    s = _table(bench, "subjects")
    subjects = {str(row["subject_id"]): {k: _s(row.get(k)) for k in SUBJECT_KEYS}
                for row in s.to_dict("records")}
    it = _table(bench, "items", ["item_id", "content", "item_features"])
    items = {str(i): (_s(c), _s(f)) for i, c, f in
             zip(it.item_id.to_numpy(), it.content.to_numpy(), it.item_features.to_numpy())}
    b = _table(bench, "benchmarks")
    if b is not None and len(b):
        dom = b.iloc[0].get("domain")
        meta["domain"] = str(list(dom)[0]) if dom is not None and len(dom) else "unknown"
    return BenchData(bench, subjects, items, r, meta)


def pair_stats():
    """Per (benchmark, subject): distinct items, responses, correct. Cached; used by fitting."""
    path = os.path.join(CACHE, "pair_stats.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    rows = []
    for bench in eligible_benchmarks():
        r = _table(bench, "responses", ["subject_id", "item_id", "response"])
        if r is None:
            continue
        r = r[r.response.isin([0.0, 1.0])]
        if r.empty:
            continue
        g = r.groupby("subject_id")
        df = pd.DataFrame({"n_items": g.item_id.nunique(), "n_resp": g.size(),
                           "n_correct": g.response.sum()}).reset_index()
        df["benchmark"] = bench
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(path)
    return out


def benchmark_summary():
    """Domain and response count for every usable benchmark (used to stratify splits)."""
    path = os.path.join(CACHE, "benchmark_summary.json")
    if os.path.exists(path):
        return json.load(open(path))
    out = {}
    for bench in eligible_benchmarks():
        r = _table(bench, "responses", ["response"])
        if r is None or r.response.isin([0.0, 1.0]).sum() == 0:
            continue
        b = _table(bench, "benchmarks")
        dom = b.iloc[0].get("domain") if b is not None and len(b) else None
        out[bench] = {"domain": str(list(dom)[0]) if dom is not None and len(dom) else "unknown",
                      "n_responses": int(r.response.isin([0.0, 1.0]).sum())}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out
