# Experiment 068 — CCT5 + UMAP-50 + BIRCH (k=5, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_068`
- **Status:** executed — K=5 counterpart of exp_061
- **Varies vs. exp_061:** n_clusters [4] → [5] (threshold sweep kept)

## Setup

CCT5 diff embeddings (66 × 768-dim, Kaggle run), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[5], cosine
silhouette, equal-depth (5 bins). Sandbox: `/tmp/opencode/s03run_fx5/g68/`.

Artifacts: `../models/exp_068_best_clustering.pkl`, `../figures/exp_068_umap_birch_k5.png`.

## 2. Results (k = 4 → 5 → 6)

| k | Winner (thr, nc) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.0225, 4) | 0.4909 | 0.1197 | 0.4194 | exp_061 |
| 5 | (0.0225, 5) | 0.5285 | **0.1332** | 0.3548 | exp_068 |
| 6 | (0.0225, 6) | 0.5347 | 0.2038 | 0.3065 | exp_027 |

- Clusters at k=5: `[8, 9, 14, 15, 16]`.
- Same threshold (0.0225) wins all three k — the cut location is stable, only
  its fineness varies.

## Reading

Gentle monotone climb (0.120 → 0.133 → 0.204) with purity falling (0.42 → 0.35 →
0.31): the cleanest bin-count trade in the study — every added bin buys NMI and
costs purity at a steady rate. CCT5 is the well-behaved middle child: no
surprises, no collapses.
