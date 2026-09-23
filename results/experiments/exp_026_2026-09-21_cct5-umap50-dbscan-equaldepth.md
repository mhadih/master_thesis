# Experiment 026 — CCT5 + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-21
- **ID:** `exp_026`
- **Status:** partial — Kaggle-generated CCT5 embeddings (see exp_025 §1)
- **Varies vs. exp_002/006/010/014/018/022:** embedding model only (→ CCT5); all else identical

## 1. Setup

Same embeddings (66 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_025. Clustering: **DBSCAN** grid (20 eps × min_samples
2…9, <4-cluster skip, cosine silhouette) via `clustering/dbscan_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_cct5/db/`. Artifacts: `../models/exp_026_best_clustering.pkl`,
`../figures/exp_026_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0256, 9)** |
| **Silhouette (cosine)** | **0.4464** |
| **Clusters** | **4 + 11 noise** (script stores best_k=5, counting noise); sizes `[8, 9, 11(noise), 14, 20]` |
| **NMI** (vs. 5 equal-depth grade bins) | **0.1564** |
| **Purity** | **0.3548** |

- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- NMI 0.1564 ≈ KMeans 0.1461 on this embedding — both weak; CCT5 joins the
  sub-0.22 band with CodeBERT/CCBERT/CC2Vec, well below CodeT5/e5.
- Noise fringe (11/62, 18%) mid-range for density methods on this space.

## 4.–5. Blocked steps / reproduce

As exp_025 (replace `km/kmeans_cc5.py` with `db/dbscan_cc5.py`).
