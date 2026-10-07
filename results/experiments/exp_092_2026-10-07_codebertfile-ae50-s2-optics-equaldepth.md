# Experiment 092 — CodeBERT whole-file + Autoencoder-50 (seed 2) + OPTICS (S03)

- **Date (UTC):** 2026-10-07
- **ID:** `exp_092`
- **Status:** executed — reducer variant, seed 2 of 3
- **Varies vs. exp_090/091:** AE seed only (identical train MSE 0.001836)

## 1. Setup

As exp_090 with AE seed 2. Sandbox logic: `/tmp/opencode/s03run_red/reducer_sweep.py`.

Artifacts: `../models/exp_092_best_clustering.pkl`.

## 2. Results

| Metric | exp_092 (AE s2) | exp_090 (s0) | exp_091 (s1) |
|---|---|---|---|
| Best params | (ms=3, xi=0.01, inf) | (ms=2, xi=0.01, inf) | (ms=4, xi=0.01, inf) |
| Silhouette | −0.0644 | −0.0781 | −0.0684 |
| Clusters | **6 + 29 noise**; sizes `[3, 3, 5, 5, 5, 12]`, noise 29 (47%) | 14 + 29 noise | 4 + 30 noise |
| NMI (6 bins) | **0.2564** | 0.4448 | 0.1609 |
| Purity | 0.4032 | 0.3387 | 0.4355 |

## 3. Component observations

- Middle seed in every way (k=6, NMI 0.26): the three seeds span the full
  granularity range (4/6/14 clusters) at flat reconstruction loss — seed picks
  the resolution, the data does not constrain it. A VAE or fixed-projection
  control would be needed before trusting any single AE number (as noted in the
  comparison doc).
