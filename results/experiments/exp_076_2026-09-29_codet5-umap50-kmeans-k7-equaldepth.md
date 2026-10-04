# Experiment 076 — CodeT5 + UMAP-50 + KMeans (k=7, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_076`
- **Status:** executed — K=7 counterpart of exp_055/062/069
- **Varies vs. exp_069:** k=6 → k=7; all else identical

## Setup

CodeT5-file embeddings (68 × 768-dim), 62 students, UMAP-50 cosine +
L2-normalize, KMeans k=7, cosine silhouette, equal-depth (7 bins).
Sandbox: `/tmp/opencode/s03run_fx7/i76/`.

Artifacts: `../models/exp_076_best_clustering.pkl`, `../figures/exp_076_umap_kmeans_k7.png`.

## Results (k = 4 → 5 → 6 → 7 → 17)

| k | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|
| 4 | 0.3889 | 0.1237 | 0.4516 | exp_055 |
| 5 | 0.4996 | 0.1273 | 0.3548 | exp_062 |
| 6 | 0.4660 | 0.1505 | 0.3548 | exp_069 |
| 7 | 0.5094 | **0.2734** | 0.3871 | exp_076 |
| 17 | 0.5179 | 0.5356 | 0.3548 | exp_001 |

- Cluster sizes at k=7: `[5, 6, 8, 10, 10, 11, 12]`.
- Grade-bin borders (7 bins): `[29.04, 50.58, 60.05, 84.57, 95.0, 100.0, 103.33]` (all K=7 runs share these).

## Reading

First real step up (0.15 → 0.27) — CodeT5's curve is flat 4→6 then climbs: the
grade structure resolves in stages, with the k=17 jump still unexplained by
4↔7 interpolation. Matches the open-k sweep value exactly (deterministic).
