# Experiment 024 — CC2Vec + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_024`
- **Status:** full pipeline — fresh CC2Vec embeddings (see exp_021 §1)
- **Varies vs. exp_004/008/012/016/020:** embedding model only (→ CC2Vec); all else identical

## 1. Setup

Same embeddings (65 users × 196-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_021. Clustering: **OPTICS** (`cluster_method='xi'`) grid
(min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via
`clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_cc2v/op/`. Artifacts: `../models/exp_024_best_clustering.pkl`,
`../figures/exp_024_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(6, 0.01, inf)** |
| **Silhouette (cosine)** | **0.1511** |
| **Clusters** | **4 + 24 noise** (script stores best_k=4, counting noise); sizes `[7, 9, 9, 13, 24(noise)]` — 39% noise, highest in the series |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0442** |
| **Purity** | **0.3710** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- Silhouette 0.1511 and 39% noise: OPTICS finds essentially no density structure —
  the negative extreme of the series, consistent with KMeans/DBSCAN/BIRCH readings
  of this space.
- OPTICS stays the most embedding-sensitive algorithm (NMI 0.04–0.55); CC2Vec joins
  CodeBERT at its low end.

## 4.–5. Blocked steps / reproduce

As exp_021 (replace `km/kmeans_cc2v.py` with `op/optics_cc2v.py`).
