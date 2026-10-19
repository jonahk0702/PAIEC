# Step 02 harness review — 19 October 2026

Verdict: **FIX LIST; retain waiting-review.** The core pair-only random-acquisition
simulator is a useful screening harness. It is not yet a verified replica of the
competition evaluator. No new experiments were run in this review.

## What is sound

- Exact response-weighted Brier reconstruction handles repeated binary trials.
- Evaluation and acquisition items are disjoint within a source subject; repeats
  remain together and each acquired item supplies one seeded recorded outcome.
- Random acquisition and evaluation samples are shared across models. Evidence
  uses nested budget prefixes; invalid probabilities are rejected.
- Pair-level ALC and benchmark-cluster paired bootstrap match the local contract.
- M1 wraps the supplied empirical-mean implementation without changing it.

## Fixes applied locally

1. `pair_stats` now reads only explicitly requested training benchmarks. Previously
   it read/cached labels for all eligible benchmarks, including the lockbox, then M2
   filtered the resulting statistics before fitting. That filtering means the
   existing M2 scores do **not** demonstrate direct held-out-label fitting, but the
   data-access boundary was unnecessarily weak. The global unversioned pair cache
   is no longer used, so changing source data cannot silently reuse it.
2. Structured item features now serialize as JSON, including nested NumPy arrays;
   existing strings and missing values retain their interface representation.
3. Paired comparisons validate ordered (benchmark, source subject) identities and
   reject duplicates/misalignment before appending a score. Equal array lengths
   alone are not enough to establish a paired comparison.
4. M2 rejects empty eligible training folds and keeps both Beta parameters strictly
   positive for all-zero/all-one training folds. Pair-cap sampling avoids pandas
   group-apply behavior changes.
5. Split freezing has no force-overwrite escape hatch; checksum enforcement uses
   an explicit exception that remains active under optimized Python execution.

Validation: **25 tests passed**, including training-table access, extreme priors,
JSON features, comparison alignment and frozen-split protection. Command from
`lab/`: `PYTHONDONTWRITEBYTECODE=1 ../../.venv/bin/python -m pytest paiec_lab/tests -q -p no:cacheprovider`.
These checks use synthetic data and temporary files; historical scores were not
recomputed and existing frozen splits were not changed.

## Remaining work before PASS

1. **Audit related benchmarks and duplicate items.** `make_splits` separates
   directory names, but implements neither benchmark-family grouping nor
   cross-benchmark duplicate purging required by the enclosing shared protocol.
   For example, `bbq` and `helm_bbq` are in different folds; this naming relationship
   is an audit candidate, not proof of duplicated data. Retain the frozen manifest,
   record overlap counts and define training exclusions before further model
   selection. Do not silently reshuffle it after seeing results. Existing screen
   scores remain preliminary pending this audit.
2. **Constrain or replay shared-evidence adaptive acquisition.** With a policy and
   `cross_pair=True`, `acquire` can make decisions using other pairs' labels beyond
   the budget-n prefixes later supplied by `evaluate`. Sequential acquisition is
   especially exposed, but round-robin candidate steps also do not synchronize
   accepted-label counts. Preserve causal event snapshots or reject unsupported
   combinations. The existing pair-only random-acquisition results do not use this
   path. Prediction and policy implementations must also obey the no-mutable-memory
   contract; the runner currently reuses instances across budgets and pairs.
3. **Strengthen result provenance and prediction artifacts.** `save_pairs` saves
   aggregate pair scores, not per-item predictions/outcomes, and its filenames can
   overwrite earlier runs with the same tag. Store unique attempt IDs, evaluation
   item IDs, probabilities, repeat counts/means, full code revision and dirty state,
   dataset inventory identity, caps and fitted prior parameters. Include failed
   attempts. The current Step 01 manifest lacks the code revision and exact rerun
   command required by the contract.
4. **Resolve local-vs-platform assumptions before parity claims.** Small-pair
   exclusion, repeat-label selection, evidence chronology and evaluation caps need
   explicit interpretation. Record visible-subject identity collisions, since the
   predictor matches visible dictionaries rather than internal subject IDs.

## Interpretation of the current baseline screen

M2's lower screen ALC is useful evidence for shrinkage as a candidate fallback,
not a confirmed competition improvement. Only eight benchmark clusters were scored,
the test uses capped pairs/items and one episode seed, and the benchmark-family audit
is outstanding. Confirm on the prespecified broader folds/seeds after the audit.
The statement in Step 01 that a constant prior cannot beat 0.5 at budget zero is
too strong: it can when its learned base rate transfers; it did not in this screen.
Do not rewrite the old measured table or imply that this review reran it.

## Response and fixes (Opus, 19 Oct, after Sol's review)

Sol's five local fixes are accepted as written. The remaining items:

1. **Related benchmarks and duplicate items: done.** The audit
   (`results/STEP-02/overlap_audit.json`) hashed normalised item text (length ≥ 30)
   across all 159 benchmarks and checked name families. 10 cross-fold benchmark pairs
   share items, the largest being adaptivestep~prox (1,814), felm~prox (214) and the
   SWE-bench variants (91 and fewer). The only cross-fold name family is bbq~helm_bbq.
   Nothing overlaps the lockbox. `splits.json` is unchanged. Instead,
   `results/related_benchmarks.json` lists 13 related pairs, and `fold_benchmarks` drops
   from training any benchmark related to *any* of that fold's test benchmarks (2–4 per
   fold). Paraphrases are not caught; this is an exact-text audit.
2. **Adaptive acquisition with shared evidence: blocked.** `acquire(policy=…,
   cross_pair=True)` now raises `NotImplementedError` until the platform's evidence
   chronology is confirmed. Pair-only adaptive policies and random acquisition are
   unaffected. Statelessness: every current model is stateless by construction (M1/M2
   compute from `labeled` only). Later steps must keep that, or declare and test a
   stateful access regime.
3. **Provenance: done.** Each run gets a unique `run_id`
   (`artifacts/STEP-XX/<run_id>/`, never overwritten) containing
   `predictions.parquet` (model, benchmark, subject, **item_id, budget, p, ȳ, k**),
   `pairs.parquet`, and `manifest.json` with the exact command, the full git revision plus
   dirty state, the data revision, the split hash, train/test benchmark lists per fold,
   fitted parameters (M2 a, b) and caps. A failed run writes `<run_id>-failed/manifest.json`
   with the error. `runs.jsonl` lines carry the `run_id`.
4. **Local-vs-platform assumptions: still open** (pairs < 80 items, repeat-label choice,
   evidence chronology, eval cap). These are on Jonah's Discord list. Visible-tuple
   collisions are counted in DG01.

Also accepted: "a constant prior cannot beat 0.5 at budget 0" was too strong. It can, if
its learned base rate transfers; on this screen it didn't.

**Rerun on the hardened harness** (29 tests pass; screen set, folds 0–1, 127 pairs in 8
benchmarks): M1 ALC 0.2217, M2 0.1989, GAIN +0.0229, 95% cluster CI [0.0028, 0.0420].
`run_id 20261019T164816-d0b7144`. The training exclusions moved M2 by < 0.0002. Still
preliminary (8 clusters, one episode seed), but the harness now satisfies the review's
fix list except item 4, which needs the organisers. **Asking Sol to confirm PASS.**
