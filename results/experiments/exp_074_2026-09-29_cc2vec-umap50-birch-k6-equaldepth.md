# Experiment 074 — CC2Vec + UMAP-50 + BIRCH (k=6, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_074`
- **Status:** executed — K=6 counterpart of exp_060/067 (identical setup to exp_023)
- **Varies vs. exp_067:** n_clusters [5] → [6] (threshold sweep kept)

## Setup

CC2Vec change embeddings (65 × 196-dim), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[6], cosine
silhouette, equal-depth (6 bins). Sandbox: `/tmp/opencode/s03run_fx6/h74/`.

Artifacts: `../models/exp_074_best_clustering.pkl`, `../figures/exp_074_umap_birch_k6.png`.

## Results (k = 4 → 5 → 6)

| k | Winner (thr, nc) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (0.005, 4) | 0.2846 | 0.0408 | 0.3548 | exp_060 |
| 5 | (0.0125, 5) | 0.2888 | 0.0740 | 0.2903 | exp_067 |
| 6 | (0.0125, 6) | 0.3070 | **0.1148** | 0.3065 | exp_074 = exp_023 |

- Clusters: `[2, 4, 13, 13, 14, 16]` (note the 2- and 4-student splinters).
- Bit-identical to exp_023 (same setup, same data, fixed seeds): third
  determinism control alongside exp_058 and the BIRCH grid of exp_027/h75.

## Reading

Monotone climb continues (0.041 → 0.074 → 0.115) — same shape as CodeBERT,
same floor-level absolute values. k cannot rescue a signal-poor space.
