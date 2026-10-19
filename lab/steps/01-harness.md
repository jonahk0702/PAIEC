# STEP 01: Harness (the most important step: correctness over speed)

**Goal:** a local simulator of the competition that every later step uses. Read
`lab/SPEC.md` §1–§5 fully before writing code.

## Code layout (create `lab/paiec_lab/`)
- `data.py`: load eligible benchmarks; build `subject` and `item` dicts exactly per
  SPEC §2; `benchmark_id` hashing; skip all-NaN benchmarks.
- `splits.py`: lockbox, 5 folds, screen set and pair cap per SPEC §3. Write
  `lab/results/splits.json` (benchmark names per fold/lockbox/screen and a sha256 of
  the file contents). **Once written and committed, never regenerate it.**
- `protocol.py`: `make_pairs(test_benchmarks)`, `run_acquisition(pair, predictor,
  policy=None)`, `evidence_at(pair, n, cross_pair=False)`. Follow SPEC §4 to the letter.
- `metrics.py`: exact aggregated Brier per SPEC §5, pair ALC, headline and secondary
  means, `paired_cluster_bootstrap(gain_per_pair, benchmark_of_pair, n=2000, seed=0)`, ECE.
- `models/baselines.py`: M0, M1 (wrap `empirical_mean/model.py` unchanged), M2.
- `run.py`: CLI `python -m paiec_lab.run --step 01 --model M1 --fold 0 --screen` that
  writes the `runs.jsonl` line and per-pair predictions to `lab/artifacts/`.
- `tests/test_harness.py` (pytest), with these tests:
  1. M0 gives exactly 0.25 at every budget and ALC = 0.25 (tolerance 1e-12).
  2. On a hand-made toy pair (3 items, one with 3 repeats), the aggregated Brier equals
     the explicit per-response Brier.
  3. Evidence at n is a prefix of the evidence at the next budget, for every pair.
  4. No evaluation-pool item ever appears in that subject's evidence.
  5. Same seed twice gives identical ALC.
  6. M1 at n = 1 equals the single label (0 or 1) and at n = 0 equals 0.5.
  7. The splits are disjoint, and no test-fold benchmark appears in any training artefact.

## Run
- `pytest lab/paiec_lab/tests -q`: everything must pass.
- M0, M1, M2 on the screen set (2 folds). Record the three ALC rows.
- **Event trace:** for one benchmark with 2 subjects, write
  `lab/results/STEP-01/trace.md`: per budget, which labels each subject sees, under
  cross-pair `off` and `on`.

## Done when
All tests pass, M0 = 0.2500 exactly, `splits.json` committed, trace written.
Then update STATUS, commit and push. **STEP 02 is a review gate: stop there.**
