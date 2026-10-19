# SPEC: the contract every step follows

Competition: Predictive AI Evaluation Competition (PAIEC), NeurIPS 2026. Final
deadline **Fri 30 Oct 2026, 23:59 AoE**. Rules: https://aims.stanford.edu/competition

## 1. The task
For one **subject** (an AI system) and one **item** (a benchmark question), return
P(correct) ∈ [0, 1]. The submitted `model.py` exposes `predict(input, labeled) -> float`:
- `input = [subject, item]` (two dicts; **every value is a string; missing = ""**)
  - subject keys: `normalized_name, provider, release_date, access_date, harness,
    harness_version, reasoning_effort, subject_features_extra`
  - item keys: `item_content, item_features, interactors, benchmark_id` (benchmark_id is
    an anonymous stable hash)
- `labeled` = list of `[[subject, item], label]`, label ∈ {0, 1}. Empty at budget 0.
- Optional `labeling.py: acquisition_function(input, prediction, labeled, context) -> bool`.
  `context = {subject_id, benchmark_id, labels_acquired, labels_remaining, max_labels: 31,
  items_remaining}`.

Test benchmarks are **never seen in training** (benchmark-level cold start).

## 2. Data
- `lab/data/measurement-db/<bench>/formatted_tables/{benchmarks,subjects,items,responses}.parquet`,
  HF dataset `aims-foundations/measurement-db`, **revision
  `54074453da7b9911be0f48e4c560be547bc84d21`**. Download with `lab/scripts/fetch_tables.py`.
  Never download `traces`.
- Eligible = `benchmarks.response_type == "binary"` (163 dirs). 4 have all-NaN responses
  (chartmuseum, ineqmath, mmmu_dev_val, naturalreasoning) and are skipped → **159 usable**.
- Known facts (do not recompute unless a step asks): 13,023 pairs with ≥80 items in 145
  benchmarks; 77% of those pairs sit in 8 benchmarks (alignment_faking 4,278 and sib200
  3,341 pairs); 51% of subjects have no `normalized_name`; `reasoning_effort` and
  `interactors` are almost always empty.
- Build input dicts exactly as in §1: subject fields from `subjects.parquet`;
  `item_content` = `items.content`; `item_features` = JSON string of `items.item_features`;
  `interactors` from the response row; `benchmark_id` = first 12 hex chars of
  sha1(benchmark dir name). NaN → "".

## 3. Splits (frozen once in step 01, committed as `lab/results/splits.json`)
- **Lockbox:** about 10% of usable benchmarks (seed 1), stratified by first `domain` tag.
  **Never** used before the final step. No exceptions.
- **Folds:** the remaining benchmarks go into 5 GroupKFold folds by benchmark (seed 0),
  stratified by domain and size. Test benchmarks of a fold contribute nothing to that
  fold's training: no items, no responses, no fitted statistics.
- **Screen set:** 24 development benchmarks (seed 2, spread over domains and sizes), 2
  folds. Used for fast screening in steps 03–06.
- **Pair cap:** when evaluating, sample at most 20 pairs per test benchmark (seed 3).
  Report both capped and uncapped means when cheap.

## 4. Protocol simulator (one pair = one subject on one test benchmark)
1. A pair is scored only if it has ≥ 80 distinct items (log how many are excluded).
2. Split the pair's items 50/50 into acquisition and evaluation pools, seeded by
   `sha1(subject_id + bench)`. Repeated trials of an item stay together.
3. Shuffle the acquisition items once, with a per-pair seed: this is the candidate stream.
4. **Acquisition pass:** walk the stream. Default (no policy) = accept every candidate
   until 31 labels. With a policy: call it with the current prediction and context;
   accept → reveal **one** recorded trial (seeded random if repeats) and append
   `[[subject, item], label]`. Stop at 31 labels or when the stream ends. If
   `items_remaining <= labels_remaining`, force-accept.
5. Budgets n ∈ {0, 1, 3, 7, 15, 31}: the pair's evidence = its **first n** acquired labels
   (nested prefixes).
6. **Cross-pair mode** (switch): `off` = `labeled` holds only this pair's labels; `on` =
   also other sampled pairs' first-n labels on the same benchmark. Default `off`; report
   `on` only where a step asks. Pair order for `on`: interleaved round-robin.
7. Predict once per distinct evaluation item; repeats reuse the prediction. Evaluation
   items are capped at 400 per pair (seeded) for speed.
8. **Never** let an evaluation-pool label reach `labeled` for that subject.

## 5. Metric
- Per pair and budget: B_n = mean over all recorded evaluation responses of (p − y)².
  Aggregate exactly: an item with k trials and mean ȳ contributes k[(p − ȳ)² + ȳ(1 − ȳ)].
- Pair ALC = 0.1·B0 + 0.2·B1 + 0.2·B3 + 0.2·B7 + 0.2·B15 + 0.1·B31.
- **Headline = unweighted mean over pairs.** Secondary = mean over benchmarks of the
  per-benchmark mean.
- Uncertainty: a paired **benchmark-cluster bootstrap** (2,000 resamples) of
  GAIN = ALC(comparator) − ALC(candidate). Positive GAIN means the candidate is better.
- Diagnostics: ECE (10 bins), NLL (clip 1e-6).

## 6. Shared models
- **M0** constant 0.5. **M1** official empirical mean (`empirical_mean/model.py`).
- **M2** Beta shrinkage: p = (a + k)/(a + b + n), with (a, b) fit by moments on
  training-fold pair accuracies, **at most 20 pairs per training benchmark** (otherwise
  two giant benchmarks set the prior). Balance every fit by benchmark like this.
- **M3** L2 logistic regression on subject features + item embedding (step 03 defines
  them).
- **M4** amortized Rasch: η0 = h(x_s) + r_s − g(e_i), with r_s a shrunk per-name
  offset (0 if no name) and g a ridge or small-MLP map from item embedding to
  difficulty.
- **O** (the update used by everything): δ ~ N(0, τ²); likelihood Π_j σ(η0_j + δ)^y_j
  (1 − σ(η0_j + δ))^(1−y_j) over the pair's labels; predict E[σ(η0_i + δ) | labels] by
  Gauss–Hermite quadrature. τ ∈ {0, .25, .5, 1, 2}, chosen on inner training-benchmark
  episodes. Check 41 vs 81 nodes agree to 1e-4.

## 7. Decision rule
**Advance** if GAIN ≥ 0.001 and the 95% bootstrap interval is above 0. **Demote** if
the interval's upper end is below 0.001. Otherwise **inconclusive**: say the cheapest
next test. Never tune on the lockbox. At most 6 configurations per model family per step.

## 8. What every step writes
- `lab/results/STEP-XX/RESULT.md`: decision paragraph, a table (B0 B1 B3 B7 B15 B31 ALC
  GAIN CI), verdict, deviations from the step.
- `lab/results/STEP-XX/manifest.json`: data revision, split hash, configs tried, seeds,
  git SHA, wall time, artifact and log paths.
- Append to `lab/results/runs.jsonl`, one line per run:
  `{step, card, variant, fold, seed, alc, b0, b1, b3, b7, b15, b31, ece, gain, ci_lo, ci_hi, n_pairs, n_benchmarks, wall_s, git_sha}`
- Big files (embeddings, predictions, logs) go in `lab/artifacts/` (git-ignored); the
  manifest points to them.
- Never commit data, credentials, zips or model weights.
