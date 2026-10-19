# STEP 03: Cheap frontier and the first submission artefact (cards DG03, PR01, DG02 noise)

**Goal:** know how far simple models get, and produce a zip Jonah can submit.

## Do
1. **Embeddings:** `BAAI/bge-small-en-v1.5` (sentence-transformers) on `item_content`
   (truncate to 512 tokens). For training, use at most 3,000 items per benchmark
   (seed 4). Cache to `lab/artifacts/emb/<bench>.npy`. Record the model revision.
2. **Subject features:** one-hot provider (top 30 + other), harness (top 50 + other),
   release year (numeric, missing flag), has_name flag, hashed name (2^12 buckets), and
   parsed parameter count (regex `(\d+(\.\d+)?)\s*[bB]\b`, log, missing flag).
3. **Models** on the screen set, 2 folds, cross-pair `off`, random acquisition:
   M1, M2, M3 (logistic on subject features + embedding, C ∈ {0.1, 1}), and M3+O.
   Weight training rows so each training pair has equal total weight.
4. **References (not submittable):** the in-sample pair rate and the benchmark rate on
   evaluation responses. Report them as context only.
5. **Local variability:** M1 and M2 on 50 resampled formative-sized samples (8 pairs,
   about 120 items each). Report the SD of ALC and the SD of the paired M2 − M1
   difference.
6. **Artefacts for Jonah:** build `dist/empirical_mean.zip` with `python tools/build_mean_zip.py` and a
   `dist/m2.zip` (M2 in `model.py`, fitted (a, b) as constants). Check both with
   `python check_submission_zip.py dist/<zip>`. Do **not** submit them; Jonah does.

## Outputs
RESULT.md with the table (M1, M2, M3, M3+O, references), the variability numbers, the
zip check output and the embedding throughput (items/s).

## Done when
The table is filled, both zips pass `check_submission_zip.py`, and the runs are in `runs.jsonl`.
