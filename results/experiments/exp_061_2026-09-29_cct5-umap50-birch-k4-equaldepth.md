# Experiment 061 — CCT5 + UMAP-50 + BIRCH (k=4, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_061`
- **Status:** executed — fixed-K rerun of exp_027 (n_clusters=4)
- **Varies vs. exp_027:** n_clusters [4,5,6,8,10] → fixed [4] (threshold sweep kept)

## 1. Setup

CCT5 diff embeddings (66 × 768-dim, Kaggle run), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[4], cosine
silhouette, equal-depth (4 bins). Sandbox: `/tmp/opencode/s03run_fx/f61/`.

Artifacts: `../models/exp_061_best_clustering.pkl`, `../figures/exp_061_umap_birch_k4.png`.

## 2. Results

| Metric | exp_061 (k=4) | exp_027 (k=6) |
|---|---|---|
| Best params | **(thr=0.0225, nc=4)** | (thr=0.0225, nc=6) |
| Silhouette | 0.4909 | 0.5347 |
| Clusters | **4, no noise**; sizes `[9, 14, 15, 24]` | 6, no noise |
| NMI (4 bins) | **0.1197** | 0.2038 (6 bins) |
| Purity | **0.4194** | 0.3065 |

## 3. Component observations

- Same threshold (0.0225), coarser cut: NMI 0.204 → 0.120 with purity rising
  (0.31 → 0.42) — the standard bin-count trade, mid-sized among the seven.
  CCT5 stays top of the change-model band at fixed k.
