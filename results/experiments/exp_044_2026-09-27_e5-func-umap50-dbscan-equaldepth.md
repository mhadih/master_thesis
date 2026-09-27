# Experiment 044 — E5 function-level + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_044`
- **Status:** full pipeline — DB-generated function-level E5 embeddings (see exp_043 §1)
- **Varies vs. exp_006:** embedding protocol only (whole-file → function-level); all else identical

## 1. Setup

Same embeddings (67 users × 768-dim), students (62, user 143 absent), UMAP-50 cosine
+ L2-normalize, and equal-depth eval as exp_043. Clustering: **DBSCAN** grid
(20 eps × min_samples 2…9, <4-cluster skip, cosine silhouette) via
`clustering/dbscan_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_ef/db/`. Artifacts: `../models/exp_044_best_clustering.pkl`,
`../figures/exp_044_umap_dbscan.png`.

## 2. Results

| Metric | exp_044 (function-level) | exp_006 (whole-file) |
|---|---|---|
| Best params | **(eps≈0.0210, ms=6)** | (eps≈0.0206, ms=5) |
| Silhouette | 0.5737 | 0.4688 |
| Clusters | **5 + 2 noise** (stored k=6, counting noise); sizes `[2, 8, 8, 9, 13, 22]` | 4 + 8 noise |
| NMI | 0.1919 | 0.1354 |
| Purity | 0.3548 | 0.3871 |

- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- Same operating point (eps≈0.021, ms≈5–6), better compactness (0.57 vs. 0.47),
  slightly better NMI (0.19 vs. 0.14) — but both weak. Density structure sharpens
  without becoming grade-relevant: function splitting concentrates E5 space into
  tighter, still grade-orthogonal cores.
- Thinnest noise fringe in the E5 pair (2 vs. 8).

## 4.–5. Blocked steps / reproduce

As exp_043 (replace `km/kmeans_ef.py` with `db/dbscan_ef.py`).
