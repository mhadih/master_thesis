# Experiment 014 — GraphCodeBERT + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_014`
- **Status:** full pipeline — fresh GraphCodeBERT embeddings (see exp_013 §1)
- **Varies vs. exp_002/006/010:** embedding model only (→ GraphCodeBERT); all else identical

## 1. Setup

Same embeddings, students (62), UMAP-50 cosine + L2-normalize, and equal-depth eval
as exp_013. Clustering: **DBSCAN** grid (20 eps × min_samples 2…9, <4-cluster skip,
cosine silhouette) via `clustering/dbscan_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_gcb/db/`. Artifacts: `../models/exp_014_best_clustering.pkl`,
`../figures/exp_014_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0246, 8)** |
| **Silhouette (cosine)** | **0.6551** |
| **Clusters** | **3 + 12 noise** (script stores best_k=4, counting noise); sizes `[10, 12(noise), 15, 25]` |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0927** |
| **Purity** | **0.4032** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- **Identical partition to BIRCH/exp_015:** ARI(DBSCAN, BIRCH) = 1.0 — DBSCAN's
  12-point "noise" is exactly BIRCH's 4th cluster.

## 3. Component observations (for the conclusion)

- NMI 0.0927 ≈ BIRCH 0.0927, KMeans 0.0932 — under GraphCodeBERT all three agree the
  space has ~4 groups with no grade signal. Cross-check with CodeBERT (0.09–0.21):
  GraphCodeBERT behaves like CodeBERT, not like CodeT5/e5.
- Purity 0.4032 matches KMeans exactly (same 4 bins; similar coarse structure).

## 4.–5. Blocked steps / reproduce

As exp_013 (replace `km/kmeans_gcb.py` with `db/dbscan_gcb.py`).
