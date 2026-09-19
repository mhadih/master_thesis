# Experiment 018 — CCBERT + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_018`
- **Status:** full pipeline — fresh CCBERT embeddings (see exp_017 §1)
- **Varies vs. exp_002/006/010/014:** embedding model only (→ CCBERT); all else identical

## 1. Setup

Same embeddings (66 users × 512-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_017. Clustering: **DBSCAN** grid (20 eps × min_samples
2…9, <4-cluster skip, cosine silhouette) via `clustering/dbscan_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_ccb/db/`. Artifacts: `../models/exp_018_best_clustering.pkl`,
`../figures/exp_018_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0178, 4)** |
| **Silhouette (cosine)** | **0.7130** |
| **Clusters** | **4, no noise**; sizes `[7, 15, 16, 24]` |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0637** |
| **Purity** | **0.3548** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- **Identical partition to BIRCH/exp_019** (ARI = 1.0) — second exact DBSCAN≡BIRCH
  agreement in the series (cf. GraphCodeBERT exp_014/015).

## 3. Component observations (for the conclusion)

- NMI 0.0637 is the series floor (tied with BIRCH on this embedding): density
  grouping finds compact (sil 0.71) but grade-orthogonal structure.
- DBSCAN≡BIRCH repeats across two embedding models now (GraphCodeBERT, CCBERT) but
  not CodeT5/e5/CodeBERT — agreement itself is embedding-dependent; worth a line.

## 4.–5. Blocked steps / reproduce

As exp_017 (replace `km/kmeans_ccb.py` with `db/dbscan_ccb.py`).
