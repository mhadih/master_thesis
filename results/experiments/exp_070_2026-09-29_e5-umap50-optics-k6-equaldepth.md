# Experiment 070 — e5 + UMAP-50 + OPTICS (6 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_070`
- **Status:** executed — K=6 counterpart of exp_056/063
- **Varies vs. exp_063:** exactly-5 → exactly-6 non-noise clusters; all else identical

## Setup

e5 embeddings (68 × 768-dim), 62 students, UMAP-50 cosine + L2-normalize, OPTICS
`xi` grid constrained to 6 non-noise clusters (noise allowed), cosine
silhouette, equal-depth (6 bins). Sandbox: `/tmp/opencode/s03run_fx6/h70/`.

Artifacts: `../models/exp_070_best_clustering.pkl`, `../figures/exp_070_umap_optics_k6.png`.

## Results (k = 4 → 5 → 6 → 17)

| k | Winner (ms, xi) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (7, 0.01) | 0.3856 | 0.1106 | 0.4355 | exp_056 |
| 5 | (5, 0.01) | 0.2664 | 0.1905 | 0.4194 | exp_063 |
| 6 | (4, 0.1) | 0.2026 | **0.1658** | 0.3387 | exp_070 |
| 17 | (2, 0.01) | 0.4603 | 0.5485 | 0.3710 | exp_008 |

- Clusters at k=6: 6 + 30 noise `[4, 4, 5, 6, 6, 7]`, noise 30 (48% — the fringe
  keeps growing with k).

## Reading

NMI dips (0.19 → 0.17) while noise balloons (19 → 30): forcing finer density
cuts on e5 manufactures noise faster than structure. The k=5→6 step already
shows the fine-granularity NMI (0.55 at k=17) is built on increasingly fragile
partitions — relevant context when citing exp_008's headline number.
