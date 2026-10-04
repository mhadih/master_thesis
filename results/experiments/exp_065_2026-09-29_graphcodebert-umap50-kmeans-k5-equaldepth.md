# Experiment 065 — GraphCodeBERT + UMAP-50 + KMeans (k=5, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_065`
- **Status:** executed — K=5 counterpart of exp_058
- **Varies vs. exp_058:** k=4 → k=5; all else identical

## Setup

GraphCodeBERT function embeddings (67 × 768-dim), 62 students, UMAP-50 cosine +
L2-normalize, KMeans k=5, cosine silhouette, equal-depth (5 bins).
Sandbox: `/tmp/opencode/s03run_fx5/g65/`.

Artifacts: `../models/exp_065_best_clustering.pkl`, `../figures/exp_065_umap_kmeans_k5.png`.

## Results

| Metric | exp_065 (k=5) | exp_058 (k=4) |
|---|---|---|
| Silhouette | 0.6065 | 0.6540 |
| Clusters | `[10, 11, 12, 14, 15]` | `[10, 14, 15, 23]` |
| NMI (5 bins) | **0.1156** | 0.0932 (4 bins) |
| Purity | 0.3710 | 0.4032 |

## Reading

Gentle NMI rise (0.093 → 0.116) with falling silhouette (0.65 → 0.61): one more
split buys a little grade information at some compactness cost — the same
direction as CodeBERT, at a tenth of the magnitude. Protocol-insensitive model,
k-mildly-sensitive.
