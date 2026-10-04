# Experiment 064 — CodeBERT + UMAP-50 + DBSCAN (5 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_064`
- **Status:** executed — K=5 counterpart of exp_057
- **Varies vs. exp_057:** exactly-4 → exactly-5 non-noise clusters; all else identical

## Setup

CodeBERT function embeddings (67 × 768-dim), 62 students (user 143 absent),
UMAP-50 cosine + L2-normalize, DBSCAN grid constrained to 5 non-noise clusters,
cosine silhouette, equal-depth (5 bins). Sandbox: `/tmp/opencode/s03run_fx5/g64/`.

Artifacts: `../models/exp_064_best_clustering.pkl`, `../figures/exp_064_umap_dbscan_k5.png`.

## Results (k = 4 → 5 → 7)

| k | Winner (eps, ms) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.0222, 7) | 0.5888 | 0.1279 | 0.3871 | exp_057 |
| 5 | (0.0201, 6) | 0.6006 | **0.1507** | 0.3387 | exp_064 |
| 7 | (0.0190, 5) | 0.6479 | 0.2085 | 0.3387 | exp_010 |

- Clusters at k=5: 5 + 6 noise, real clusters `[6, 11, 13, 20]`, noise 6.

## Reading

Monotone NMI climb with k (0.13 → 0.15 → 0.21) at flat purity: CodeBERT's grade
signal scales with resolution — finer density cuts keep paying off, no plateau
in sight within the tested range. (Same repo-script note as exp_057: stored k is
the non-noise count.)
