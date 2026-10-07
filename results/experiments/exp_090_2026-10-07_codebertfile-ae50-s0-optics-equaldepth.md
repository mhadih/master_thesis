# Experiment 090 — CodeBERT whole-file + Autoencoder-50 (seed 0) + OPTICS (S03)

- **Date (UTC):** 2026-10-07
- **ID:** `exp_090`
- **Status:** executed — reducer variant, seed 0 of 3 (see reducer_comparison_dim50.md)
- **Varies vs. exp_038:** reducer only (UMAP-50 → MLP autoencoder 768-256-64-50, MSE, full-batch 200 epochs, train MSE 0.001836); all else identical

## 1. Setup

CodeBERT whole-file (68 × 768-dim), 62 students, AE latent (seed 0) +
L2-normalize, OPTICS `xi` grid, cosine silhouette, equal-depth (bins = 14).
Sandbox logic: `/tmp/opencode/s03run_red/reducer_sweep.py`.

Artifacts: `../models/exp_090_best_clustering.pkl`.

## 2. Results

| Metric | exp_090 (AE s0) | exp_038 (UMAP) |
|---|---|---|
| Best params | (ms=2, xi=0.01, inf) | (ms=2, xi=0.01, inf) |
| Silhouette | −0.0781 | 0.3903 |
| Clusters | **14 + 29 noise**; sizes `[2×10, 3×3, 4]`, noise 29 (47%) | 20 + 6 noise |
| NMI (14 bins) | **0.4448** | 0.6641 |
| Purity | 0.3387 | 0.4194 |

## 3. Component observations

- Best non-UMAP number in the sweep (0.44) from fourteen 2–4-student micro-groups:
  the AE latent shatters into fine fragments that happen to track grades — same
  fine-granularity mechanism as exp_038's k=20, without any compactness (sil < 0).
- Single-seed number: report only inside the mean±std (0.287±0.117); never alone.
