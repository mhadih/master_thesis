# Experiment 071 — CodeBERT + UMAP-50 + DBSCAN (6 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_071`
- **Status:** executed — K=6 counterpart of exp_057/064
- **Varies vs. exp_064:** exactly-5 → exactly-6 non-noise clusters; all else identical

## Setup

CodeBERT function embeddings (67 × 768-dim), 62 students (user 143 absent),
UMAP-50 cosine + L2-normalize, DBSCAN grid constrained to 6 non-noise clusters,
cosine silhouette, equal-depth (6 bins). Sandbox: `/tmp/opencode/s03run_fx6/h71/`.

Artifacts: `../models/exp_071_best_clustering.pkl`, `../figures/exp_071_umap_dbscan_k6.png`.

## Results (k = 4 → 5 → 6 → 7)

| k | Winner (eps, ms) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.0222, 7) | 0.5888 | 0.1279 | 0.3871 | exp_057 |
| 5 | (0.0201, 6) | 0.6006 | 0.1507 | 0.3387 | exp_064 |
| 6 | (0.0190, 5) | 0.6479 | **0.1937** | 0.3548 | exp_071 |
| 7 | (0.0190, 5) | 0.6479 | 0.2085 | 0.3387 | exp_010 |

- Clusters at k=6: 6 + 3 noise, real clusters `[4, 6, 6, 7, 15, 21]`, noise 3.
- Same operating point as exp_010 (eps≈0.0190, ms=5): the fixed-6 grid rediscovers
  the open-k winner's neighborhood — no fallback needed.

## Reading

Steady monotone climb (0.128 → 0.151 → 0.194 → 0.209) with the thinnest noise
fringe in the pair (3): CodeBERT density structure refines cleanly — the
best-behaved k-curve in the study. (Same repo-script note as exp_057: stored k
is the non-noise count.)
