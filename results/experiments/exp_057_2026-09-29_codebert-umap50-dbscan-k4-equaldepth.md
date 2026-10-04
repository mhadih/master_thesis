# Experiment 057 — CodeBERT + UMAP-50 + DBSCAN (4 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_057`
- **Status:** executed — fixed-K rerun of exp_010 (exactly 4 non-noise clusters)
- **Varies vs. exp_010:** open grid (winner 6+3n) → constrained to 4 non-noise clusters

## 1. Setup

CodeBERT function embeddings (67 × 768-dim), 62 students (user 143 absent),
UMAP-50 cosine + L2-normalize, DBSCAN grid (20 eps × ms 2…9), cosine silhouette,
equal-depth (4 bins). Sandbox: `/tmp/opencode/s03run_fx/f57/`.

Artifacts: `../models/exp_057_best_clustering.pkl`, `../figures/exp_057_umap_dbscan_k4.png`.

## 2. Results

| Metric | exp_057 (k=4) | exp_010 (k=7) |
|---|---|---|
| Best params | **(eps≈0.0222, ms=7)** | (eps≈0.0190, ms=5) |
| Silhouette | 0.5888 | 0.6479 |
| Clusters | **4 + 9 noise**; sizes `[6, 6, 20, 21]`, noise 9 | 6 + 3 noise |
| NMI (4 bins) | **0.1279** | 0.2085 (7 bins) |
| Purity | **0.3871** | 0.3387 |

- Note: repo script stores total labels incl. noise as best_k; fixed here to the
  non-noise count (4) so eval bins = 4 like all fixed-K runs.

## 3. Component observations

- Mild NMI drop (0.21 → 0.13), purity up (0.34 → 0.39): CodeBERT loses less than
  CodeT5/e5 under coarse graining — its grade signal, though weak, is less
  resolution-dependent.
