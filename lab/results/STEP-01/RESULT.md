# STEP 01: Harness (written by Opus, 19 Oct 2026)

## Decision
The simulator, metric and splits are built and all 17 harness tests pass. On real data,
M0 gives exactly 0.2500 at every budget, M1 gives 0.2500 at budget 0, no evaluation item
was ever acquired, and the evidence prefixes are nested (see `trace.md`). On the screen
set (folds 0 and 1, 127 pairs in 8 benchmarks), **Beta shrinkage (M2) beats the official
mean (M1) by 0.023 ALC, 95% cluster CI [0.003, 0.042]: advance M2 as the working
fallback.** The gain comes from budgets 1–3: M1 jumps to 0 or 1 after one label
(B1 = 0.321, worse than 0.5), and M2 shrinks it (B1 = 0.224).

## Table (screen set, cross-pair off, random acquisition)
| Model | fold | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ALC (bench-avg) | GAIN vs M1 | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | pooled | .2500 | .2500 | .2500 | .2500 | .2500 | .2500 | .2500 | .2500 | −.0283 | [−.084, .021] |
| M1 | pooled | .2500 | .3213 | .2188 | .1852 | .1738 | .1690 | .2217 | .2414 | — | — |
| M2 | pooled | .2570 | .2237 | .2022 | .1827 | .1737 | .1689 | **.1991** | .2106 | **+.0227** | **[.003, .042]** |
| M2 | 0 | .2563 | .2228 | .2109 | .1926 | .1820 | .1779 | .2051 | .2215 | +.0257 | [.010, .054] |
| M2 | 1 | .2578 | .2246 | .1924 | .1717 | .1646 | .1589 | .1923 | .1923 | +.0193 | [−.027, .046] |

## Verdict
Harness: **ready for review (step 02).** M2: **advance** as the fallback and the
comparator for step 03. M2's B0 (0.257) is slightly worse than 0.5. A constant prior
can't beat 0.5 at budget 0 when test base rates vary. That is the job of the item- and
subject-aware priors in steps 03–04.

## Deviations and notes
- **M2 fit fixed during the step:** the first fit weighted all training pairs equally,
  so `alignment_faking` (4,278 pairs) or `sib200` (3,341) set the prior (mean 0.48 /
  0.64), and B0 was 0.28–0.31. Now it samples at most 20 pairs per training benchmark,
  the same cap as evaluation. **Lesson for every later fit: balance training by
  benchmark.**
- Evaluation items are capped at 400 per pair (seeded). This is now in SPEC §4.
- Two benchmarks marked `release: withheld` upstream are in the pool:
  `imagenet_hard` (lockbox) and `visual_memory` (screen). They are kept because the
  splits are frozen; flag them in any report table.
- Pairs with fewer than 80 items are excluded (16 in fold 0, 48 in fold 1). How the
  official platform treats them is still an open Discord question.
- The screen set has only 8 benchmarks with scored pairs in folds 0–1, so the CIs are
  wide. Use the full folds for confirmation.
- Run locally on the design Mac (`PAIEC_DATA` pointed at the existing download), not on
  the server.
