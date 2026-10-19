# STEP 04: Amortized Rasch and the Bayesian offset update (cards PR02, UP01)

**Goal:** the cheap backbone everything else builds on.

## Do
1. **M4** per SPEC §6: g = ridge (α ∈ {1, 10}) from embedding to item difficulty, with
   difficulty first fitted by a Rasch model on training responses. h = ridge on subject
   features. r_s = per-name offset shrunk by λ ∈ {1, 10}. Variant (b): fit h, g, r jointly
   end to end with logistic loss (one small MLP for g: 384→128→1).
2. **O update** on top of M3 and M4: τ ∈ {0, .25, .5, 1, 2}, chosen per prior on inner
   training-benchmark episodes (hold out 20% of training benchmarks inside the fold).
   Check quadrature 41 vs 81 nodes, including all-0 and all-1 label prefixes.
3. Compare on the screen set, 2 folds: M2, M3+O, M4(ridge)+O, M4(joint)+O. Report the
   GAIN of each against M2 with the bootstrap CI.
4. Report cohorts: named-seen / named-new / anonymous subjects.
5. Package the best one as `dist/best04.zip` (fitted parameters as files inside the zip;
   embedding model via `models.txt` prefetch, or bundled if under 200 MB). Check it with
   `check_submission_zip.py`.

## Done when
The table and cohorts are written, the τ chosen per prior is recorded, the zip passes the
check, and the quadrature check passes.
