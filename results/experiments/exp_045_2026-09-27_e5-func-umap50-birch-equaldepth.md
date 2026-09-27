# Experiment 045 — E5 function-level + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_045`
- **Status:** full pipeline — DB-generated function-level E5 embeddings (see exp_043 §1)
- **Varies vs. exp_007:** embedding protocol only (whole-file → function-level); all else identical

## 1. Setup

Same embeddings (67 users × 768-dim), students (62, user 143 absent), UMAP-50 cosine
+ L2-normalize, and equal-depth eval as exp_043. Clustering: **BIRCH** grid
(thresholds [0.005, 0.025] step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine
silhouette) via post-fix `clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_ef/bi/`. Artifacts: `../models/exp_045_best_clustering.pkl`,
`../figures/exp_045_umap_birch.png`.

## 2. Results

| Metric | exp_045 (function-level) | exp_007 (whole-file) |
|---|---|---|
| Best params | **(thr=0.0225, nc=4)** | (thr=0.0075, nc=10) |
| Silhouette | 0.5712 | 0.5197 |
| Clusters | **4, no noise**; sizes `[13, 16, 16, 17]` | 10, no noise |
| NMI | 0.0919 | 0.3273 |
| Purity | 0.4194 | 0.3226 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- Fixed threshold grid worked unmodified (winner 0.0225, near grid top) — seventh
  embedding model validating the exp_003 rescaling.

## 3. Component observations (for the conclusion)

- NMI collapses 0.327 → 0.092 while the winner coarsens (10 → 4 clusters):
  function-level E5 admits only a coarse 4-way split with no grade content —
  mirrors the KMeans reading (exp_043).
- Winner at grid top edge again (0.0225, as in CCT5/exp_027): whole/function spaces
  here are more spread than CodeT5/e5-function predecessors.

## 4.–5. Blocked steps / reproduce

As exp_043 (replace `km/kmeans_ef.py` with `bi/birch_ef.py`).
