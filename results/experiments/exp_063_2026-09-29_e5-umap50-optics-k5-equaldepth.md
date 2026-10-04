# Experiment 063 — e5 + UMAP-50 + OPTICS (5 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_063`
- **Status:** executed — K=5 counterpart of exp_056
- **Varies vs. exp_056:** exactly-4 → exactly-5 non-noise clusters; all else identical

## Setup

e5 embeddings (68 × 768-dim), 62 students, UMAP-50 cosine + L2-normalize, OPTICS
`xi` grid constrained to 5 non-noise clusters (noise allowed), cosine silhouette,
equal-depth (5 bins). Sandbox: `/tmp/opencode/s03run_fx5/g63/`.

Artifacts: `../models/exp_063_best_clustering.pkl`, `../figures/exp_063_umap_optics_k5.png`.

## 2. Results (k = 4 → 5 → 17)

| k | Winner (ms, xi) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (7, 0.01) | 0.3856 | 0.1106 | 0.4355 | exp_056 |
| 5 | (5, 0.01) | 0.2664 | **0.1905** | 0.4194 | exp_063 |
| 17 | (2, 0.01) | 0.4603 | 0.5485 | 0.3710 | exp_008 |

- Clusters at k=5: 5 + 19 noise `[5, 7, 8, 9, 14]`, noise 19 (31% — heaviest fringe yet).

## Reading

NMI rises (0.11 → 0.19) while silhouette *falls* (0.39 → 0.27): the 5th cluster
splits off grade-relevant structure from the noise fringe at the cost of
compactness — opposite to the KMeans/CodeT5 pattern where finer k only diluted.
Density methods and centroid methods respond to k in opposite directions here.
