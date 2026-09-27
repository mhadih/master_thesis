# Experiment 040 — GraphCodeBERT whole-file + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_040`
- **Status:** full pipeline — DB-generated whole-file GraphCodeBERT embeddings (see exp_039 §1)
- **Varies vs. exp_014:** embedding protocol only (function-level → whole-file); all else identical

## 1. Setup

Same embeddings (68 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_039. Clustering: **DBSCAN** grid (20 eps × min_samples
2…9, <4-cluster skip, cosine silhouette) via `clustering/dbscan_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_gbf/db/`. Artifacts: `../models/exp_040_best_clustering.pkl`,
`../figures/exp_040_umap_dbscan.png`.

## 2. Results

| Metric | exp_040 (whole-file) | exp_014 (function-level) |
|---|---|---|
| Best params | **(eps≈0.0236, ms=9)** | (eps≈0.0230, ms=7) |
| Silhouette | 0.5825 | 0.6551 |
| Clusters | **3 + 18 noise** (stored k=4, counting noise); sizes `[10, 16, 18, 18(noise)]` | 3 + 12 noise |
| NMI | 0.1161 | 0.0927 |
| Purity | **0.4516** | 0.4032 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- Small uniform uplift (NMI 0.116 vs. 0.093, purity 0.45 vs. 0.40) at the cost of a
  heavier noise fringe (18 vs. 12): whole-file space is slightly more grade-aligned
  but more diffuse at the margins.
- Same eps neighborhood, higher min_samples (9 vs. 7) — consistent with a denser
  core + sparser halo structure.

## 4.–5. Blocked steps / reproduce

As exp_039 (replace `km/kmeans_gbf.py` with `db/dbscan_gbf.py`).
