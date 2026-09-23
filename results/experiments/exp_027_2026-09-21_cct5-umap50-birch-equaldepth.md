# Experiment 027 — CCT5 + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-21
- **ID:** `exp_027`
- **Status:** partial — Kaggle-generated CCT5 embeddings (see exp_025 §1)
- **Varies vs. exp_003/007/011/015/019/023:** embedding model only (→ CCT5); all else identical

## 1. Setup

Same embeddings (66 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_025. Clustering: **BIRCH** grid (thresholds [0.005, 0.025]
step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_cct5/bi/`. Artifacts: `../models/exp_027_best_clustering.pkl`,
`../figures/exp_027_umap_birch.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.0225, 6)** |
| **Silhouette (cosine)** | **0.5347** |
| **Clusters** | **6, no noise**; sizes `[4, 8, 9, 11, 14, 16]` |
| **NMI** (vs. 6 equal-depth grade bins) | **0.2038** |
| **Purity** | **0.3065** |

- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]`.
- Fixed threshold grid worked unmodified (winner 0.0225, near grid top) — fourth
  embedding model validating the exp_003 rescaling.

## 3. Component observations (for the conclusion)

- Best NMI on this embedding (0.2038) but still the low band: BIRCH's forced 6-way
  split (incl. a 4-student splinter) over-partitions, same reading as CC2Vec/exp_023.
- CCT5 does not rescue BIRCH-vs-grades alignment either.

## 4.–5. Blocked steps / reproduce

As exp_025 (replace `km/kmeans_cc5.py` with `bi/birch_cc5.py`).
