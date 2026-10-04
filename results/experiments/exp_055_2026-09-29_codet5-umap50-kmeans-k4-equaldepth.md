# Experiment 055 — CodeT5 + UMAP-50 + KMeans (k=4, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_055`
- **Status:** executed — fixed-K rerun of exp_001 (K=4, see fixed-K rationale in `fixed_k_comparison.md`)
- **Varies vs. exp_001:** k sweep 4…20 → fixed k=4; all else identical

## 1. Setup

CodeT5-file embeddings (68 users × 768-dim), 62 graded students, UMAP-50 cosine +
L2-normalize, KMeans k=4 (`random_state=42`), cosine silhouette, equal-depth
(4 bins). Sandbox: `/tmp/opencode/s03run_fx/f55/`.

Artifacts: `../models/exp_055_best_clustering.pkl`, `../figures/exp_055_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_055 (k=4) | exp_001 (k=17) |
|---|---|---|
| Silhouette | 0.3889 | 0.5179 |
| NMI (4 bins) | **0.1237** | 0.5356 (17 bins) |
| Purity | **0.4516** | 0.3548 |

- Cluster sizes: `[10, 14, 17, 21]`.
- Grade-bin borders: `[50.17, 78.83, 98.92, 103.33]` (all fixed-K runs share these).

## 3. Component observations

- NMI collapses 0.54 → 0.12 while purity *rises* (0.35 → 0.45): textbook
  bin-count effect — coarse bins are easy majority-votes but carry little mutual
  information. This pair alone justifies the fixed-K redesign.
