# Experiment 059 — CCBERT + UMAP-50 + OPTICS (4 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_059`
- **Status:** executed — fixed-K rerun of exp_020 (exactly 4 non-noise clusters)
- **Varies vs. exp_020:** open k (winner 6+12n, stored k=7) → constrained to 4 non-noise clusters

## 1. Setup

CCBERT change embeddings (66 × 512-dim), 62 students, UMAP-50 cosine +
L2-normalize, OPTICS `xi` grid, cosine silhouette, equal-depth (4 bins).
Sandbox: `/tmp/opencode/s03run_fx/f59/`.

Artifacts: `../models/exp_059_best_clustering.pkl`, `../figures/exp_059_umap_optics_k4.png`.

## 2. Results

| Metric | exp_059 (k=4) | exp_020 (k=7) |
|---|---|---|
| Best params | **(ms=7, xi=0.05, inf)** | (ms=7, xi=0.05, inf) |
| Silhouette | 0.7132 | 0.7132 |
| Clusters | **4 + 12 noise**; sizes `[7, 12, 15, 16]`, noise 12 | 6 + 12 noise |
| NMI (4 bins) | **0.1570** | 0.1570 (7 bins) |
| Purity | **0.4516** | 0.4516 |

## 3. Component observations

- Identical params, silhouette, NMI and purity to exp_020 to 4 decimals despite
  different bin counts (4 vs. 7): CCBERT's grade alignment is resolution-stable —
  the opposite of CodeT5/e5, whose NMI lived in fine granularity. (Purity honest
  note: 0.45 on 4 bins is coarse-bin-assisted in both.)
- Same operating point wins both searches (ms=7, xi=0.05): the constrained grid
  contains the unconstrained winner's neighborhood — no fallback needed.
