# Experiment 052 — CCT5 flow-conditioned + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_052`
- **Status:** partial — Kaggle-generated flow embeddings (see exp_051 §1)
- **Varies vs. exp_026:** embedding view only (diff-string → DFG-conditioned); all else identical

## 1. Setup

Same embeddings (65 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_051. Clustering: **DBSCAN** grid (20 eps × min_samples
2…9, <4-cluster skip, cosine silhouette) via `clustering/dbscan_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_cct5f/db/`. Artifacts: `../models/exp_052_best_clustering.pkl`,
`../figures/exp_052_umap_dbscan.png`.

## 2. Results

| Metric | exp_052 (flow view) | exp_026 (diff view) |
|---|---|---|
| Best params | **(eps≈0.0221, ms=7)** | (eps≈0.0256, ms=9) |
| Silhouette | 0.4981 | 0.4464 |
| Clusters | **3 + 9 noise** (script stores best_k=4, counting noise); sizes `[9, 15(noise), 15, 23]` | 4 + 11 noise |
| NMI | 0.1003 | 0.1564 |
| Purity | 0.3710 | 0.3548 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- Same verdict as KMeans: tighter groups (0.50 vs. 0.45), slightly lower NMI
  (0.10 vs. 0.16) — flow conditioning densifies without informing.
- Same eps neighborhood, lighter min_samples (7 vs. 9): flow space cores form
  marginally easier.

## 4.–5. Blocked steps / reproduce

As exp_051 (replace `km/kmeans_cf.py` with `db/dbscan_cf.py`).
