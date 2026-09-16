# Experiment 015 — GraphCodeBERT + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_015`
- **Status:** full pipeline — fresh GraphCodeBERT embeddings (see exp_013 §1)
- **Varies vs. exp_003/007/011:** embedding model only (→ GraphCodeBERT); all else identical

## 1. Setup

Same embeddings, students (62), UMAP-50 cosine + L2-normalize, and equal-depth eval
as exp_013. Clustering: **BIRCH** grid (thresholds [0.005, 0.025] step 0.0025 ×
n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_gcb/bi/`. Artifacts: `../models/exp_015_best_clustering.pkl`,
`../figures/exp_015_umap_birch.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.0225, 4)** |
| **Silhouette (cosine)** | **0.6551** |
| **Clusters** | **4, no noise**; sizes `[10, 12, 15, 25]` |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0927** |
| **Purity** | **0.4032** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- **Identical partition to DBSCAN/exp_014** (ARI = 1.0): hierarchical and density
  views agree completely here — rare across the series.

## 3. Component observations (for the conclusion)

- Fixed-threshold grid worked out of the box (winner 0.0225 inside [0.005, 0.025]) —
  validates the exp_003 rescaling on a second embedding model.
- NMI 0.0927 ties series-lowest: third independent algorithm confirming no grade
  structure in GraphCodeBERT space at this granularity.

## 4.–5. Blocked steps / reproduce

As exp_013 (replace `km/kmeans_gcb.py` with `bi/birch_gcb.py`).
