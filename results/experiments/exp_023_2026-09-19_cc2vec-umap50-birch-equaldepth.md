# Experiment 023 — CC2Vec + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_023`
- **Status:** full pipeline — fresh CC2Vec embeddings (see exp_021 §1)
- **Varies vs. exp_003/007/011/015/019:** embedding model only (→ CC2Vec); all else identical

## 1. Setup

Same embeddings (65 users × 196-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_021. Clustering: **BIRCH** grid (thresholds [0.005, 0.025]
step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_cc2v/bi/`. Artifacts: `../models/exp_023_best_clustering.pkl`,
`../figures/exp_023_umap_birch.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.0125, 6)** |
| **Silhouette (cosine)** | **0.3070** |
| **Clusters** | **6, no noise**; sizes `[2, 4, 13, 13, 14, 16]` |
| **NMI** (vs. 6 equal-depth grade bins) | **0.1148** |
| **Purity** | **0.3065** |

- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]`.
- Fixed threshold grid worked unmodified (winner mid-grid) — third embedding model
  validating the exp_003 rescaling.

## 3. Component observations (for the conclusion)

- Best NMI on this embedding (0.1148) yet still near floor; BIRCH's forced 6-way
  split (incl. 2- and 4-student splinters) looks like over-partitioning noise.
- No DBSCAN≡BIRCH agreement (cf. GraphCodeBERT/CCBERT): unstructured space →
  algorithm disagreement, itself a diagnostic worth one line.

## 4.–5. Blocked steps / reproduce

As exp_021 (replace `km/kmeans_cc2v.py` with `bi/birch_cc2v.py`).
