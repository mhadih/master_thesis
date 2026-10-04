# Experiment 080 — CCBERT + UMAP-50 + OPTICS (7 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_080`
- **Status:** executed — K=7 counterpart of exp_059/066/073
- **Varies vs. exp_073:** exactly-6 → exactly-7 non-noise clusters; all else identical

## Setup

CCBERT change embeddings (66 × 512-dim), 62 students, UMAP-50 cosine +
L2-normalize, OPTICS `xi` grid constrained to 7 non-noise clusters (noise
allowed), cosine silhouette, equal-depth (7 bins). Sandbox: `/tmp/opencode/s03run_fx7/i80/`.

Artifacts: `../models/exp_080_best_clustering.pkl`, `../figures/exp_080_umap_optics_k7.png`.

## Results (k = 4 → 5 → 6 → 7 → 7open)

| k | Winner (ms, xi) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (7, 0.05) | 0.7132 | 0.1570 | 0.4516 | exp_059 |
| 5 | (7, 0.01) | 0.6982 | 0.1833 | 0.4355 | exp_066 |
| 6 | (5, 0.05) | 0.5563 | 0.2180 | 0.3871 | exp_073 |
| 7 | (3, 0.1) | 0.0255 | **0.2822** | 0.3871 | exp_080 |
| 7 | (7, 0.05) | 0.7132 | 0.1570 | 0.4516 | exp_020 (open) |

- Clusters at k=7: 7 + 27 noise `[3, 3, 3, 5, 5, 7, 9]`, noise 27 (44% — min_samples=3
  fragments aggressively).

## Reading

NMI keeps climbing (0.22 → 0.28, CCBERT's best anywhere) while silhouette
collapses (0.56 → 0.03): the grade-aligned partition here is a *loose*
constellation of micro-clusters plus a huge noise fringe — the exact inverse of
the compactness-≠-relevance pattern seen elsewhere. Report both numbers together
or neither: 0.28 without 0.03 misleads, 0.03 without 0.28 undersells.
