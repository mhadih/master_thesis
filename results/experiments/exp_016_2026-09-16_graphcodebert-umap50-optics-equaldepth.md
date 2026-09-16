# Experiment 016 — GraphCodeBERT + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_016`
- **Status:** full pipeline — fresh GraphCodeBERT embeddings (see exp_013 §1)
- **Varies vs. exp_004/008/012:** embedding model only (→ GraphCodeBERT); all else identical

## 1. Setup

Same embeddings, students (62), UMAP-50 cosine + L2-normalize, and equal-depth eval
as exp_013. Clustering: **OPTICS** (`cluster_method='xi'`) grid (min_samples 2…9 ×
xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via `clustering/optics_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_gcb/op/`. Artifacts: `../models/exp_016_best_clustering.pkl`,
`../figures/exp_016_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(8, 0.05, inf)** |
| **Silhouette (cosine)** | **0.6275** |
| **Clusters** | **3 + 11 noise** (script stores best_k=3, counting noise); sizes `[10, 11(noise), 14, 27]` |
| **NMI** (vs. 3 equal-depth grade bins) | **0.0922** |
| **Purity** | **0.5161** |

- Grade-bin borders (3 bins): `[55.39, 95.0, 103.33]` — identical to exp_012 (same k).
- NMI/purity treat noise as an ordinary label; purity 0.52 on 3 bins is a coarseness
  artifact (cf. NMI ≈ 0.09), same as exp_012.

## 3. Component observations (for the conclusion)

- NMI 0.0922 ≈ series floor alongside BIRCH/DBSCAN/KMeans on this embedding (all
  ≈0.09): all four algorithms independently agree GraphCodeBERT space carries
  essentially no grade signal — the strongest negative result in the series.
- Winner at (8, 0.05): between CodeBERT's degenerate (5, 0.2) and e5's fine (2, 0.01)
  operating points — OPTICS remains the most embedding-sensitive algorithm
  (NMI range 0.09–0.55).

## 4.–5. Blocked steps / reproduce

As exp_013 (replace `km/kmeans_gcb.py` with `op/optics_gcb.py`).
