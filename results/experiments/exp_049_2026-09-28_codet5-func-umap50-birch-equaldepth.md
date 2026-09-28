# Experiment 049 — CodeT5 function-level + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_049`
- **Status:** full pipeline — DB-generated function-level CodeT5 embeddings (see exp_047 §1)
- **Varies vs. exp_003:** embedding protocol only (whole-file → function-level); all else identical

## 1. Setup

Same embeddings (67 users × 768-dim), students (62, user 143 absent), UMAP-50 cosine
+ L2-normalize, and equal-depth eval as exp_047. Clustering: **BIRCH** grid
(thresholds [0.005, 0.025] step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine
silhouette) via post-fix `clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_ct5f/bi/`. Artifacts: `../models/exp_049_best_clustering.pkl`,
`../figures/exp_049_umap_birch.png`.

## 2. Results

| Metric | exp_049 (function-level) | exp_003 (whole-file) |
|---|---|---|
| Best params | **(thr=0.0125, nc=4)** | (thr=0.01, nc=10) |
| Silhouette | 0.6781 | 0.5172 |
| Clusters | **4, no noise**; sizes `[8, 17, 18, 19]` | 10, no noise |
| NMI | 0.0556 | 0.3073 |
| Purity | 0.3226 | 0.2903 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- Fixed threshold grid worked unmodified (winner mid-grid) — eighth embedding
  model validating the exp_003 rescaling.

## 3. Component observations (for the conclusion)

- NMI collapses 0.307 → 0.056 with coarsening (10 → 4 clusters): function-level
  CodeT5 admits only a coarse grade-free split — same verdict as KMeans/DBSCAN
  on this embedding.
- Compactness up (0.68 vs. 0.52): the recurring function-level signature
  (tighter groups, less signal).

## 4.–5. Blocked steps / reproduce

As exp_047 (replace `km/kmeans_ct5f.py` with `bi/birch_ct5f.py`).
