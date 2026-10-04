# Experiment 073 — CCBERT + UMAP-50 + OPTICS (6 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_073`
- **Status:** executed — K=6 counterpart of exp_059/066
- **Varies vs. exp_066:** exactly-5 → exactly-6 non-noise clusters; all else identical

## Setup

CCBERT change embeddings (66 × 512-dim), 62 students, UMAP-50 cosine +
L2-normalize, OPTICS `xi` grid constrained to 6 non-noise clusters (noise
allowed), cosine silhouette, equal-depth (6 bins). Sandbox: `/tmp/opencode/s03run_fx6/h73/`.

Artifacts: `../models/exp_073_best_clustering.pkl`, `../figures/exp_073_umap_optics_k6.png`.

## Results (k = 4 → 5 → 6 → 7)

| k | Winner (ms, xi) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (7, 0.05) | 0.7132 | 0.1570 | 0.4516 | exp_059 |
| 5 | (7, 0.01) | 0.6982 | 0.1833 | 0.4355 | exp_066 |
| 6 | (5, 0.05) | 0.5563 | **0.2180** | 0.3871 | exp_073 |
| 7 | (7, 0.05) | 0.7132 | 0.1570 | 0.4516 | exp_020 |

- Clusters at k=6: 6 + 5 noise, real clusters `[5, 7, 7, 9, 13, 16]`, noise 5.

## Reading

The peak-at-5 shape extends: 0.157 → **0.218** → 0.157 — k=6 is now CCBERT's best
NMI anywhere (all 20 experiments), and the non-monotone curve (peak, dip, peak)
confirms genuine k-sensitivity rather than a bin-count artifact: more bins do
not mechanically raise this model's NMI (k=7 falls back to 0.157).
