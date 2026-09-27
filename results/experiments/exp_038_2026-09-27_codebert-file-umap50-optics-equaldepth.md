# Experiment 038 — CodeBERT whole-file + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_038`
- **Status:** partial — Kaggle-generated whole-file CodeBERT embeddings (see exp_035 §1)
- **Varies vs. exp_012:** embedding protocol only (function-level → whole-file); all else identical

## 1. Setup

Same embeddings (68 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_035. Clustering: **OPTICS** (`cluster_method='xi'`) grid
(min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via
`clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_cbf/op/`. Artifacts: `../models/exp_038_best_clustering.pkl`,
`../figures/exp_038_umap_optics.png`.

## 2. Results

| Metric | exp_038 (whole-file) | exp_012 (function-level) |
|---|---|---|
| Best params | **(ms=2, xi=0.01, inf)** | (ms=5, xi=0.2, inf) |
| Silhouette | 0.3903 | 0.5728 |
| Clusters | **20 + 6 noise** (stored k=20); sizes `[2×8, 3×6, 4×5, 6]` | 2 + 28 noise |
| NMI | **0.6641** | 0.1539 |
| Purity | 0.4194 | 0.5000 |

- Grade-bin borders (20 bins): `[22.33, 27.12, 31.17, 40.38, 50.17, 52.75, 55.67,
  60.05, 66.33, 78.83, 84.93, 86.17, 94.17, 96.7, 98.92, 100.0, 101.25, 101.5,
  101.83, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- **NMI 0.6641 is the series best by a wide margin** (prev. best: OPTICS/e5 0.5485)
  — but it comes with k=20 micro-clusters for 62 students (~3 each) against 20
  grade bins (~3.1 each). At this granularity NMI rewards near-1:1 cluster-bin
  matching, so treat 0.66 as an upper-bound demonstration of structure, not as a
  directly comparable score: the fair comparison is the fixed-bin variant (open
  item since exp_002).
- Protocol contrast is extreme: function-level OPTICS collapsed (2 clusters + 45%
  noise, NMI 0.15) while whole-file fragments into 20 fine groups — whole-file
  space has rich fine structure that function averaging washed out.
- OPTICS remains the most embedding-sensitive algorithm (NMI now spans
  0.04–0.66).

## 4.–5. Blocked steps / reproduce

As exp_035 (replace `km/kmeans_cbf.py` with `op/optics_cbf.py`).
