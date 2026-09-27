# Experiment 041 — GraphCodeBERT whole-file + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_041`
- **Status:** full pipeline — DB-generated whole-file GraphCodeBERT embeddings (see exp_039 §1)
- **Varies vs. exp_015:** embedding protocol only (function-level → whole-file); all else identical

## 1. Setup

Same embeddings (68 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_039. Clustering: **BIRCH** grid (thresholds [0.005, 0.025]
step 0.0025 × n_clusters [4, 5, 6, 8, 10], cosine silhouette) via post-fix
`clustering/birch_clustering.py`.

Run sandbox: `/tmp/opencode/s03run_gbf/bi/`. Artifacts: `../models/exp_041_best_clustering.pkl`,
`../figures/exp_041_umap_birch.png`.

## 2. Results

| Metric | exp_041 (whole-file) | exp_015 (function-level) |
|---|---|---|
| Best params | **(thr=0.025, nc=4)** | (thr=0.01, nc=10) |
| Silhouette | 0.5895 | 0.6551 |
| Clusters | **4, no noise**; sizes `[10, 16, 17, 19]` | 10, no noise |
| NMI | 0.0932 | 0.0927 |
| Purity | 0.4194 | 0.4032 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- Fixed threshold grid worked unmodified (winner at grid top edge again — sixth
  embedding model validating the exp_003 rescaling).

## 3. Component observations (for the conclusion)

- NMI identical to 4 decimals (0.0932 vs. 0.0927) with different winners
  (k=4 vs. k=10): BIRCH finds no more grade signal either way — protocol-invariant
  null result, the cleanest in the pair series.
- Coarser winner (4 vs. 10 clusters) at similar silhouette: whole-file space
  supports fewer, broader groups.

## 4.–5. Blocked steps / reproduce

As exp_039 (replace `km/kmeans_gbf.py` with `bi/birch_gbf.py`).
