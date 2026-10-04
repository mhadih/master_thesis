# Experiment 081 — CC2Vec + UMAP-50 + BIRCH (k=7, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_081`
- **Status:** executed — K=7 counterpart of exp_060/067/074
- **Varies vs. exp_074:** n_clusters [6] → [7] (threshold sweep kept)

## Setup

CC2Vec change embeddings (65 × 196-dim), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[7], cosine
silhouette, equal-depth (7 bins). Sandbox: `/tmp/opencode/s03run_fx7/i81/`.

Artifacts: `../models/exp_081_best_clustering.pkl`, `../figures/exp_081_umap_birch_k7.png`.

## Results (k = 4 → 5 → 6 → 7)

| k | Winner (thr, nc) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.005, 4) | 0.2846 | 0.0408 | 0.3548 | exp_060 |
| 5 | (0.0125, 5) | 0.2888 | 0.0740 | 0.2903 | exp_067 |
| 6 | (0.0125, 6) | 0.3070 | 0.1148 | 0.3065 | exp_074 |
| 7 | (0.0125, 7) | 0.3201 | **0.1638** | 0.3065 | exp_081 |

- Clusters at k=7: `[2, 4, 6, 10, 13, 13, 14]` (note the 2-student splinter).

## Reading

Monotone climb continues (0.041 → 0.074 → 0.115 → 0.164) — same shape as
CodeBERT, same floor-level absolute values. k cannot rescue a signal-poor
space, but the curve never saturates in range: if a future setup needs CC2Vec
squeezed further, keep widening k (with splinter warnings attached).
