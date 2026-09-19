# Experiment 019 — CCBERT + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_019`
- **Status:** full pipeline — fresh CCBERT embeddings (see exp_017 §1)
- **Varies vs. exp_003/007/011/015:** embedding model only (→ CCBERT); all else identical

## 1. Setup

Same embeddings (66 users × 512-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_017. Clustering: **BIRCH** grid (thresholds [0.005, 0.025]
step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_ccb/bi/`. Artifacts: `../models/exp_019_best_clustering.pkl`,
`../figures/exp_019_umap_birch.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.005, 4)** |
| **Silhouette (cosine)** | **0.7130** |
| **Clusters** | **4, no noise**; sizes `[7, 15, 16, 24]` |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0637** |
| **Purity** | **0.3548** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- **Identical partition to DBSCAN/exp_018** (ARI = 1.0).

## 3. Component observations (for the conclusion)

- Fixed threshold grid works out of the box again (winner 0.005 at grid edge —
  lower edge this time; if a future embedding wants <0.005 the grid must extend).
- NMI 0.0637 ties series floor: fourth independent algorithm confirming no grade
  structure in CCBERT space at this granularity.

## 4.–5. Blocked steps / reproduce

As exp_017 (replace `km/kmeans_ccb.py` with `bi/birch_ccb.py`).
