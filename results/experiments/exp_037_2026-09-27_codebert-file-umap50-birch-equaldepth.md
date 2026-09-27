# Experiment 037 — CodeBERT whole-file + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_037`
- **Status:** partial — Kaggle-generated whole-file CodeBERT embeddings (see exp_035 §1)
- **Varies vs. exp_011:** embedding protocol only (function-level → whole-file); all else identical

## 1. Setup

Same embeddings (68 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_035. Clustering: **BIRCH** grid (thresholds [0.005, 0.025]
step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_cbf/bi/`. Artifacts: `../models/exp_037_best_clustering.pkl`,
`../figures/exp_037_umap_birch.png`.

## 2. Results

| Metric | exp_037 (whole-file) | exp_011 (function-level) |
|---|---|---|
| Best params | **(thr=0.025, nc=4)** | (thr=0.0075, nc=4) |
| Silhouette | 0.5171 | 0.6828 |
| Clusters | **4, no noise**; sizes `[7, 11, 15, 29]` | [6, 12, 21, 23] |
| NMI | 0.0699 | 0.0943 |
| Purity | 0.3710 | 0.3871 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- Fixed threshold grid worked unmodified (winner 0.025 at grid top edge — fifth
  embedding model validating the exp_003 rescaling, though at the edge).

## 3. Component observations (for the conclusion)

- Protocol effect ≈ nil for BIRCH (NMI 0.070 vs. 0.094): hierarchical grouping
  finds the same coarse 4-way split either way, and neither tracks grades.
- Winner at the grid's top edge (0.025) hints whole-file space is slightly more
  spread than function space (winner 0.0075) — consistent with the lower
  silhouette (0.52 vs. 0.68).

## 4.–5. Blocked steps / reproduce

As exp_035 (replace `km/kmeans_cbf.py` with `bi/birch_cbf.py`).
