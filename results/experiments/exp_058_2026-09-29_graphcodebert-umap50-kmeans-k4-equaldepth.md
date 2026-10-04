# Experiment 058 — GraphCodeBERT + UMAP-50 + KMeans (k=4, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_058`
- **Status:** executed — fixed-K rerun of exp_013 (identical setup: exp_013 already won at k=4)
- **Varies vs. exp_013:** nothing (control reproduction)

## 1. Setup

GraphCodeBERT function embeddings (67 × 768-dim), 62 students (user 143 absent),
UMAP-50 cosine + L2-normalize, KMeans k=4, cosine silhouette, equal-depth
(4 bins). Sandbox: `/tmp/opencode/s03run_fx/f58/`.

Artifacts: `../models/exp_058_best_clustering.pkl`, `../figures/exp_058_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_058 | exp_013 |
|---|---|---|
| Silhouette | 0.6540 | 0.6540 |
| NMI | **0.0932** | 0.0932 |
| Purity | 0.4032 | 0.4032 |

- Cluster sizes: `[10, 14, 15, 23]`.

## 3. Component observations

- Bit-identical reproduction (sil/NMI/purity to 4 decimals): the pipeline is
  deterministic end-to-end (fixed seeds throughout), so all cross-exp
  differences in this thesis are real, not run noise. Keep as the determinism
  control in the methods section.
