"""Download the measurement-db formatted tables needed for PAIEC exploration.

Stage 1 (default): benchmarks + subjects tables for every directory, plus a
size inventory of items/responses/traces. Stage 2 (--tables responses items):
download those tables for competition-eligible directories (binary responses,
item granularity). Traces are never downloaded here.

    python lab/scripts/fetch_tables.py --revision <sha>
    python lab/scripts/fetch_tables.py --revision <sha> --tables responses items
"""

import argparse
import concurrent.futures as cf
import json
import os

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "measurement-db")
INVENTORY = os.path.join(HERE, "..", "data", "inventory.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tables", nargs="*", default=[])
    args = ap.parse_args()
    api = HfApi()
    if os.path.exists(INVENTORY):
        inv = json.load(open(INVENTORY))
    else:
        top = [e.path for e in api.list_repo_tree(REPO, repo_type="dataset", revision=args.revision, recursive=False)
               if "." not in e.path]
        inv = {"revision": args.revision, "dirs": {}}

        def scan(d):
            names = [f"{d}/formatted_tables/{t}.parquet" for t in
                     ("benchmarks", "subjects", "items", "responses", "response", "traces")]
            info = api.get_paths_info(REPO, names, repo_type="dataset", revision=args.revision)
            return d, {p.path.rsplit("/", 1)[1]: p.size for p in info}

        with cf.ThreadPoolExecutor(16) as ex:
            for d, sizes in ex.map(scan, top):
                inv["dirs"][d] = {"sizes": sizes}

        def meta(d):
            out = {}
            for t in ("benchmarks", "subjects"):
                if f"{t}.parquet" in inv["dirs"][d]["sizes"]:
                    hf_hub_download(REPO, f"{d}/formatted_tables/{t}.parquet", repo_type="dataset",
                                    revision=args.revision, local_dir=OUT)
            p = os.path.join(OUT, d, "formatted_tables", "benchmarks.parquet")
            if os.path.exists(p):
                b = pd.read_parquet(p).iloc[0]
                out = {"response_type": str(b.get("response_type")), "granularity": str(b.get("granularity"))}
                out["eligible"] = out["response_type"] == "binary" and out["granularity"] == "item"
            return d, out

        with cf.ThreadPoolExecutor(16) as ex:
            for d, m in ex.map(meta, list(inv["dirs"])):
                inv["dirs"][d].update(m)
        json.dump(inv, open(INVENTORY, "w"), indent=1)

    elig = [d for d, v in inv["dirs"].items() if v.get("eligible")]
    sz = lambda t: sum((inv["dirs"][d]["sizes"].get(f"{t}.parquet") or 0) for d in elig)
    print(f"{len(inv['dirs'])} dirs, {len(elig)} eligible")
    for t in ("items", "responses", "response", "traces"):
        print(f"  eligible {t}: {sz(t)/1e9:.2f} GB")

    def fetch(d):
        for t in args.tables:
            for name in ([t, "response"] if t == "responses" else [t]):
                if f"{name}.parquet" in inv["dirs"][d]["sizes"]:
                    hf_hub_download(REPO, f"{d}/formatted_tables/{name}.parquet", repo_type="dataset",
                                    revision=inv["revision"], local_dir=OUT)
                    break
        return d

    if args.tables:
        with cf.ThreadPoolExecutor(8) as ex:
            for i, _ in enumerate(ex.map(fetch, elig), 1):
                if i % 20 == 0 or i == len(elig):
                    print(f"  {i}/{len(elig)}", flush=True)


if __name__ == "__main__":
    main()
