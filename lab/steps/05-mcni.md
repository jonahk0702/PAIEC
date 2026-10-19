# STEP 05: Noise-injection uncertainty, first three tests (cards MC01, MC02, MC03)

Source: Yuan, Li, Kuruoğlu, arXiv 2501.12314. Weights become w + α·σ_l·ε with
ε ~ N(0, 1) and σ_l the std of layer l's weights, applied **in training and at
inference**. Do not perturb the frozen embedding model.

## Do
1. **MC01:** the MLP prior from step 04 (b), trained (i) without noise and (ii) with
   noise α ∈ {0.01, 0.05}. At inference draw T = 32 coherent weight samples (one sample =
   one full weight draw reused for every item in the pair); check T = 64 on one fold.
   Record whether biases are perturbed (default: no).
2. **MC02:** compare (a) the noise-trained net at mean weights, (b) the mean of the 32
   noisy predictions, and (c) the deterministic net + temperature calibration fitted on
   inner episodes. All three with the O update.
3. **MC03 (the decisive control):** condition on labels by reweighting the 32 coherent
   samples, w_t ∝ Π_j p_tj^y_j (1 − p_tj)^(1−y_j), and predict Σ_t w_t p_ti. Compare
   against a **scrambled** bank: for each distinct item, independently permute the 32
   sample indices (5 permutation seeds; the same permutation for repeated calls to that
   item). Marginals are identical, so B0 must match exactly. Report ALC and ESS = 1/Σw²
   at each budget.

## Done when
The tables for MC01, MC02 and MC03 are written, B0 is equal between coherent and
scrambled (assert this), and ESS is reported. Verdict: does coherent beat scrambled
(GAIN CI > 0)?
