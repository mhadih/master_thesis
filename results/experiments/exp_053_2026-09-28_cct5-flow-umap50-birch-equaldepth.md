# Experiment 053 — CCT5 flow-conditioned + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_053`
- **Status:** partial — Kaggle-generated flow embeddings (see exp_051 §1)
- **Varies vs. exp_027:** embedding view only (diff-string → DFG-conditioned); all else identical

## 1. Setup

Same embeddings (65 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_051. Clustering: **BIRCH** grid (thresholds [0.005, 0.025]
step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_cct5f/bi/`. Artifacts: `../models/exp_053_best_clustering.pkl`,
`../figures/exp_053_umap_birch.png`.

## 2. Results

| Metric | exp_053 (flow view) | exp_027 (diff view) |
|---|---|---|
| Best params | **(thr=0.025, nc=4)** | (thr=0.0225, nc=6) |
| Silhouette | 0.5785 | 0.5347 |
| Clusters | **4, no noise**; sizes `[5, 15, 19, 23]` | 6, no noise |
| NMI | 0.0917 | 0.2038 |
| Purity | 0.3710 | 0.3065 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- Fixed threshold grid worked unmodified (winner at grid top edge again — eighth
  embedding model validating the exp_003 rescaling).

## 3. Component observations (for the conclusion)

- Largest view gap in the CCT5 pair (NMI 0.092 vs. 0.204): hierarchical grouping
  preferred the diff view's 6-way split; the flow view collapses to a coarse
  4-way split with half the grade signal.
- Compactness up (0.58 vs. 0.53): the recurring flow signature (denser, coarser,
  no more informative).

## 4.–5. Blocked steps / reproduce

As exp_051 (replace `km/kmeans_cf.py` with `bi/birch_cf.py`).
