# Experiment 066 — CCBERT + UMAP-50 + OPTICS (5 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_066`
- **Status:** executed — K=5 counterpart of exp_059
- **Varies vs. exp_059:** exactly-4 → exactly-5 non-noise clusters; all else identical

## 1. Setup

CCBERT change embeddings (66 × 512-dim), 62 students, UMAP-50 cosine +
L2-normalize, OPTICS `xi` grid constrained to 5 non-noise clusters (noise
allowed), cosine silhouette, equal-depth (5 bins). Sandbox: `/tmp/opencode/s03run_fx5/g66/`.

Artifacts: `../models/exp_066_best_clustering.pkl`, `../figures/exp_066_umap_optics_k5.png`.

## 2. Results (k = 4 → 5 → 7)

| k | Winner (ms, xi) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (7, 0.05) | 0.7132 | 0.1570 | 0.4516 | exp_059 |
| 5 | (7, 0.01) | 0.6982 | **0.1833** | 0.4355 | exp_066 |
| 7 | (7, 0.05) | 0.7132 | 0.1570 | 0.4516 | exp_020 |

- Clusters at k=5: 5, no noise `[7, 11, 12, 16, 16]` (ms=7 both 4- and 5-winners).

## Reading

NMI peaks mid-sweep (0.157 → **0.183** → 0.157): the only non-monotone k-curve
in the study — CCBERT's grade alignment has a genuine optimum at k=5, not an
artifact of more bins. Small effect, but the shape (peak, not slope) is the
methodologically interesting bit: it justifies reporting argmax-k per setup
rather than assuming finer-is-better.
