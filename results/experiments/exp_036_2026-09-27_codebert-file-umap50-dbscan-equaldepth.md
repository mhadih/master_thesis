# Experiment 036 — CodeBERT whole-file + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_036`
- **Status:** partial — Kaggle-generated whole-file CodeBERT embeddings (see exp_035 §1)
- **Varies vs. exp_010:** embedding protocol only (function-level → whole-file); all else identical

## 1. Setup

Same embeddings (68 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_035. Clustering: **DBSCAN** grid (20 eps × min_samples
2…9, <4-cluster skip, cosine silhouette) via `clustering/dbscan_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_cbf/db/`. Artifacts: `../models/exp_036_best_clustering.pkl`,
`../figures/exp_036_umap_dbscan.png`.

## 2. Results

| Metric | exp_036 (whole-file) | exp_010 (function-level) |
|---|---|---|
| Best params | **(eps≈0.0172, ms=2)** | (eps≈0.0190, ms=5) |
| Silhouette | 0.4042 | 0.6479 |
| Clusters | **12 + 5 noise** (stored k=13); sizes `[2, 2, 2, 4, 4, 4, 5, 5, 5, 5, 6, 7, 11]` | 6 + 3 noise |
| NMI | **0.4651** | 0.2085 |
| Purity | 0.3710 | 0.3387 |

- Grade-bin borders (13 bins): `[23.33, 29.04, 40.38, 50.58, 55.39, 60.05, 70.67,
  84.57, 86.17, 95.0, 98.83, 100.0, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- **Whole-file more than doubles function-level NMI** (0.4651 vs. 0.2085) — the
  largest protocol effect in the CodeBERT pair, and the second-best NMI in the
  whole 32-experiment series (after OPTICS/codebert-file exp_038).
- Caveat: k=13 fine bins flatter the NMI (13 bins ≈ 4.8 students each); still, the
  gap to exp_010 (7 bins, 0.21) is too large for bin-count effects alone.
- min_samples=2 (chained micro-clusters) beat exp_010's ms=5: whole-file space
  fragments into small dense groups where function space formed 6 broad ones.

## 4.–5. Blocked steps / reproduce

As exp_035 (replace `km/kmeans_cbf.py` with `db/dbscan_cbf.py`).
