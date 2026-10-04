# Experiment 062 — CodeT5 + UMAP-50 + KMeans (k=5, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_062`
- **Status:** executed — K=5 counterpart of exp_055 (k-effect study with exp_001/055)
- **Varies vs. exp_055:** k=4 → k=5; all else identical

## Setup

CodeT5-file embeddings (68 × 768-dim), 62 students, UMAP-50 cosine +
L2-normalize, KMeans k=5, cosine silhouette, equal-depth (5 bins).
Sandbox: `/tmp/opencode/s03run_fx5/g62/`.

Artifacts: `../models/exp_062_best_clustering.pkl`, `../figures/exp_062_umap_kmeans_k5.png`.

## Results (k = 4 → 5 → 17)

| k | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|
| 4 | 0.3889 | 0.1237 | 0.4516 | exp_055 |
| 5 | 0.4996 | **0.1273** | 0.3548 | exp_062 |
| 17 | 0.5179 | 0.5356 | 0.3548 | exp_001 |

- Cluster sizes at k=5: `[8, 11, 13, 14, 16]`.
- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]` (all K=5 runs share these).

## Reading

NMI flat (0.124 → 0.127) while purity falls (0.45 → 0.35): one extra bin splits a
coherent majority vote without adding mutual information — the k-effect here is
purely the bin-count trade, and the k=17 outlier (0.54) remains unexplained by
4↔5 interpolation.
