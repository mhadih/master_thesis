# Experiment 048 — CodeT5 function-level + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_048`
- **Status:** full pipeline — DB-generated function-level CodeT5 embeddings (see exp_047 §1)
- **Varies vs. exp_002:** embedding protocol only (whole-file → function-level); all else identical

## 1. Setup

Same embeddings (67 users × 768-dim), students (62, user 143 absent), UMAP-50 cosine
+ L2-normalize, and equal-depth eval as exp_047. Clustering: **DBSCAN** grid
(20 eps × min_samples 2…9, <4-cluster skip, cosine silhouette) via
`clustering/dbscan_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_ct5f/db/`. Artifacts: `../models/exp_048_best_clustering.pkl`,
`../figures/exp_048_umap_dbscan.png`.

## 2. Results

| Metric | exp_048 (function-level) | exp_002 (whole-file) |
|---|---|---|
| Best params | **(eps≈0.0227, ms=8)** | (eps≈0.0230, ms=7) |
| Silhouette | 0.6627 | 0.4586 |
| Clusters | **3 + 6 noise** (stored k=4, counting noise); sizes `[6(noise), 18, 18, 20]` | 6 + 7 noise |
| NMI | 0.0455 | 0.2231 |
| Purity | 0.3226 | 0.3387 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- Same eps neighborhood, far tighter groups (0.66 vs. 0.46) with 5× less NMI
  (0.046 vs. 0.223): function splitting concentrates CodeT5 space into dense
  grade-orthogonal cores — the CodeBERT-mirror in reverse.
- Thinnest noise fringe in this pair (6 vs. 7).

## 4.–5. Blocked steps / reproduce

As exp_047 (replace `km/kmeans_ct5f.py` with `db/dbscan_ct5f.py`).
