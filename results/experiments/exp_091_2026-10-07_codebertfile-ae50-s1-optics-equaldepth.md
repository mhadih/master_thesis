# Experiment 091 — CodeBERT whole-file + Autoencoder-50 (seed 1) + OPTICS (S03)

- **Date (UTC):** 2026-10-07
- **ID:** `exp_091`
- **Status:** executed — reducer variant, seed 1 of 3
- **Varies vs. exp_090:** AE seed only (identical train MSE 0.001837)

## 1. Setup

As exp_090 with AE seed 1. Sandbox logic: `/tmp/opencode/s03run_red/reducer_sweep.py`.

Artifacts: `../models/exp_091_best_clustering.pkl`.

## 2. Results

| Metric | exp_091 (AE s1) | exp_090 (AE s0) |
|---|---|---|
| Best params | (ms=4, xi=0.01, inf) | (ms=2, xi=0.01, inf) |
| Silhouette | −0.0684 | −0.0781 |
| Clusters | **4 + 30 noise**; sizes `[7, 7, 9, 9]`, noise 30 (48%) | 14 + 29 noise |
| NMI (4 bins) | **0.1609** | 0.4448 |
| Purity | 0.4355 | 0.3387 |

## 3. Component observations

- Same reconstruction loss, wildly different geometry (k=14 vs. k=4, NMI 0.44 vs.
  0.16): the AE objective does not constrain the latent layout OPTICS sees —
  seed instability documented, not averaged away. Coarsest AE partition recovers
  the least grade signal of the three seeds.
