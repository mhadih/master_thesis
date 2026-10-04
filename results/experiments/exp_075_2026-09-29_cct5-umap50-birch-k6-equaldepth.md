# Experiment 075 — CCT5 + UMAP-50 + BIRCH (k=6, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_075`
- **Status:** executed — K=6 counterpart of exp_061/068 (identical setup to exp_027)
- **Varies vs. exp_068:** n_clusters [5] → [6] (threshold sweep kept)

## Setup

CCT5 diff embeddings (66 × 768-dim, Kaggle run), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[6], cosine
silhouette, equal-depth (6 bins). Sandbox: `/tmp/opencode/s03run_fx6/h75/`.

Artifacts: `../models/exp_075_best_clustering.pkl`, `../figures/exp_075_umap_birch_k6.png`.

## 2. Results (k = 4 → 5 → 6)

| k | Winner (thr, nc) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.0225, 4) | 0.4909 | 0.1197 | 0.4194 | exp_061 |
| 5 | (0.0225, 5) | 0.5285 | 0.1332 | 0.3548 | exp_068 |
| 6 | (0.0225, 6) | 0.5347 | **0.2038** | 0.3065 | exp_075 = exp_027 |

- Clusters: `[4, 8, 9, 11, 14, 16]`.
- Bit-identical to exp_027 (same threshold, same everything) — determinism
  holds; the k=4→5→6 climb (0.120 → 0.133 → 0.204) is real, steady, and still
  the best-behaved curve in the study.

## Reading

Same threshold wins all three k (0.0225): the cut location is stable, only its
fineness varies — NMI climbs while purity falls at a steady rate. No surprises,
no collapses: the well-behaved middle child once more.
