# Experiment 060 — CC2Vec + UMAP-50 + BIRCH (k=4, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_060`
- **Status:** executed — fixed-K rerun of exp_023 (n_clusters=4)
- **Varies vs. exp_023:** n_clusters [4,5,6,8,10] → fixed [4] (threshold sweep kept)

## 1. Setup

CC2Vec change embeddings (65 × 196-dim), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[4], cosine
silhouette, equal-depth (4 bins). Sandbox: `/tmp/opencode/s03run_fx/f60/`.

Artifacts: `../models/exp_060_best_clustering.pkl`, `../figures/exp_060_umap_birch_k4.png`.

## 2. Results

| Metric | exp_060 (k=4) | exp_023 (k=6) |
|---|---|---|
| Best params | **(thr=0.005, nc=4)** | (thr=0.0125, nc=6) |
| Silhouette | 0.2846 | 0.3070 |
| Clusters | **4, no noise**; sizes `[9, 14, 19, 20]` | 6, no noise |
| NMI (4 bins) | **0.0408** | 0.1148 (6 bins) |
| Purity | 0.3548 | 0.3065 |

## 3. Component observations

- NMI falls to 0.041, series floor alongside: at fixed coarse resolution CC2Vec
  carries essentially no grade signal — consistent with exp_023's verdict, now
  unconfounded by bin count.
