# STEP 06: Graph pilot and cross-subject diagnostics (cards DG05, GR01, GR02, DG06, DG07)

## Graph construction (fixed per training fold; never rebuilt per query)
- Anchors: M = 256 training items (balanced over training benchmarks, deduplicated), and
  M = 1024 as a second setting. Mutual kNN, k = 10, cosine on embeddings, Gaussian
  weights (bandwidth = the median kNN distance). Symmetric normalised Laplacian L.
- K_A = (I + L)^(-1) (graph Matérn, ν = 1). K_feat = the Gaussian feature kernel on the
  same anchors.
- For any item i: b(i) = normalised affinities to its 10 nearest anchors (zeros
  elsewhere). Graph kernel k(i, j) = b(i)ᵀ K_A b(j); anchor-direct kernel
  b(i)ᵀ K_feat b(j). Items farther than the 99th percentile of training kNN distance get
  b = 0 (fallback to offset only).

## Do
1. **DG05/GR01 diagnostic:** on training folds, using step 04 out-of-fold residuals
   r = y − p0, plot mean residual product against embedding-distance decile, within and
   across benchmarks, against a feature-shuffled null. Report the numbers, not just
   the plot.
2. **GR02 adaptation pilot** (screen set, 2 folds, random acquisition, cross-pair `off`),
   best step-04 prior throughout. Model: logit = η0 + δ + f(i) with δ ~ N(0, τ²) and
   f ~ GP(0, σ_g² k). Batch Laplace approximation; σ_g ∈ {0, 0.5, 1}. Variants:
   (a) offset only (σ_g = 0); (b) anchor-direct kernel; (c) graph kernel;
   (d) graph kernel on a degree-preserving rewired anchor graph. GAIN of (c) over (b) is
   the topology test; (c) over (a) is the graph-value test.
3. **DG06:** with cross-pair `on`, at each budget, what fraction of evaluation items have
   ≥1 revealed label on the same item from another subject, and on the same benchmark?
   Then compare O vs a shared benchmark offset (pooled over subjects).
4. **DG07:** distribution of acquisition-pool sizes for scored pairs (how many
   candidates can a policy skip?).

## Done when
All four reported. **Then stop.** Set STATUS to `waiting-review`: Opus writes steps 07+
from these results.
