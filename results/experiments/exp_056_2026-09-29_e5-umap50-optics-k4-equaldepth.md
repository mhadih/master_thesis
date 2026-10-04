# Experiment 056 — e5 + UMAP-50 + OPTICS (4 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_056`
- **Status:** executed — fixed-K rerun of exp_008 (exactly 4 non-noise clusters)
- **Varies vs. exp_008:** open k (winner 17+4n) → constrained grid (max silhouette
  subject to exactly 4 non-noise clusters; noise allowed as extra label)

## 1. Setup

e5 embeddings (68 × 768-dim), 62 students, UMAP-50 cosine + L2-normalize, OPTICS
`cluster_method='xi'` grid (ms 2…9 × xi [0.01, 0.05, 0.1, 0.2]), cosine
silhouette, equal-depth (4 bins). Sandbox: `/tmp/opencode/s03run_fx/f56/`.

Artifacts: `../models/exp_056_best_clustering.pkl`, `../figures/exp_056_umap_optics_k4.png`.

## 2. Results

| Metric | exp_056 (k=4) | exp_008 (k=17) |
|---|---|---|
| Best params | **(ms=7, xi=0.01, inf)** | (ms=2, xi=0.01, inf) |
| Silhouette | 0.3856 | 0.4603 |
| Clusters | **4 + 13 noise**; sizes `[11, 11, 12, 15]`, noise 13 | 17 + 4 noise |
| NMI (4 bins) | **0.1106** | 0.5485 (17 bins) |
| Purity | **0.4355** | 0.3710 |

## 3. Component observations

- Same NMI collapse as CodeT5 (0.55 → 0.11) with purity rising (0.37 → 0.44):
  e5's celebrated grade alignment lives almost entirely in fine granularity —
  at 4 bins it is indistinguishable from the pack. Strongest evidence that
  cross-model NMI must be compared at fixed k.
