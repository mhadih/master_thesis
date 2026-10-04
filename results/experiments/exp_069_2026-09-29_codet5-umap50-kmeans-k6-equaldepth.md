# Experiment 069 — CodeT5 + UMAP-50 + KMeans (k=6, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_069`
- **Status:** executed — K=6 counterpart of exp_055/062
- **Varies vs. exp_062:** k=5 → k=6; all else identical

## Setup

CodeT5-file embeddings (68 × 768-dim), 62 students, UMAP-50 cosine +
L2-normalize, KMeans k=6, cosine silhouette, equal-depth (6 bins).
Sandbox: `/tmp/opencode/s03run_fx6/h69/`.

Artifacts: `../models/exp_069_best_clustering.pkl`, `../figures/exp_069_umap_kmeans_k6.png`.

## Results (k = 4 → 5 → 6 → 17)

| k | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|
| 4 | 0.3889 | 0.1237 | 0.4516 | exp_055 |
| 5 | 0.4996 | 0.1273 | 0.3548 | exp_062 |
| 6 | 0.4660 | **0.1505** | 0.3548 | exp_069 |
| 17 | 0.5179 | 0.5356 | 0.3548 | exp_001 |

- Cluster sizes at k=6: `[7, 8, 9, 12, 12, 14]`.
- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]` (all K=6 runs share these).

## Reading

Slow monotone climb (0.124 → 0.127 → 0.151) then the k=17 jump (0.54): CodeT5
needs fine granularity; 4↔5↔6 interpolation explains nothing about the outlier.
