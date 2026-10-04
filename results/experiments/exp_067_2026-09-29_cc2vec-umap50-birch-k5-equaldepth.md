# Experiment 067 — CC2Vec + UMAP-50 + BIRCH (k=5, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_067`
- **Status:** executed — K=5 counterpart of exp_060
- **Varies vs. exp_060:** n_clusters [4] → [5] (threshold sweep kept)

## Setup

CC2Vec change embeddings (65 × 196-dim), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[5], cosine
silhouette, equal-depth (5 bins). Sandbox: `/tmp/opencode/s03run_fx5/g67/`.

Artifacts: `../models/exp_067_best_clustering.pkl`, `../figures/exp_067_umap_birch_k5.png`.

## 2. Results (k = 4 → 5 → 6)

| k | Winner (thr, nc) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.005, 4) | 0.2846 | 0.0408 | 0.3548 | exp_060 |
| 5 | (0.0125, 5) | 0.2888 | **0.0740** | 0.2903 | exp_067 |
| 6 | (0.0125, 6) | 0.3070 | 0.1148 | 0.3065 | exp_023 |

- Clusters at k=5: `[6, 13, 13, 14, 16]`.

## Reading

Monotone climb (0.041 → 0.074 → 0.115): CC2Vec behaves like CodeBERT — finer
cuts keep paying off within range. Absolute levels stay floor-bound; k cannot
rescue a signal-poor space, it can only resolve what little structure exists.
