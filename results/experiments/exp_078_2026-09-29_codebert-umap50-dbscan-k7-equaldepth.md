# Experiment 078 — CodeBERT + UMAP-50 + DBSCAN (7 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_078`
- **Status:** executed — K=7 counterpart of exp_057/064/071 (identical setup to exp_010)
- **Varies vs. exp_071:** exactly-6 → exactly-7 non-noise clusters; all else identical

## Setup

CodeBERT function embeddings (67 × 768-dim), 62 students (user 143 absent),
UMAP-50 cosine + L2-normalize, DBSCAN grid constrained to 7 non-noise clusters,
cosine silhouette, equal-depth (7 bins). Sandbox: `/tmp/opencode/s03run_fx7/i78/`.

Artifacts: `../models/exp_078_best_clustering.pkl`, `../figures/exp_078_umap_dbscan_k7.png`.

## Results (k = 4 → 5 → 6 → 7)

| k | Winner (eps, ms) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.0222, 7) | 0.5888 | 0.1279 | 0.3871 | exp_057 |
| 5 | (0.0201, 6) | 0.6006 | 0.1507 | 0.3387 | exp_064 |
| 6 | (0.0190, 5) | 0.6479 | 0.1937 | 0.3548 | exp_071 |
| 7 | (0.0179, 5) | 0.5876 | **0.2186** | 0.3387 | exp_078 |

- Clusters at k=7: 7 + 8 noise `[4, 6, 6, 7, 8, 9, 11]`, noise 8.
- Note: same (eps, ms) neighborhood as exp_010, but a different input space
  (exp_010 ran repo-default euclidean/unnormalized; this series is cosine +
  normalized) — hence NMI 0.2186 vs. 0.2085, close but not identical. Do not
  merge the two rows.

## Reading

Steady monotone climb (0.128 → 0.151 → 0.194 → 0.219) — the best-behaved k-curve
in the study, now extended: CodeBERT density structure refines cleanly at every
step, thinnest noise fringe held throughout.
