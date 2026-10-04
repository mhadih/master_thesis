# Experiment 072 — GraphCodeBERT + UMAP-50 + KMeans (k=6, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_072`
- **Status:** executed — K=6 counterpart of exp_058/065
- **Varies vs. exp_065:** k=5 → k=6; all else identical

## Setup

GraphCodeBERT function embeddings (67 × 768-dim), 62 students, UMAP-50 cosine +
L2-normalize, KMeans k=6, cosine silhouette, equal-depth (6 bins).
Sandbox: `/tmp/opencode/s03run_fx6/h72/`.

Artifacts: `../models/exp_072_best_clustering.pkl`, `../figures/exp_072_umap_kmeans_k6.png`.

## Results (k = 4 → 5 → 6)

| k | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|
| 4 | 0.6540 | 0.0932 | 0.4032 | exp_058 |
| 5 | 0.6065 | 0.1156 | 0.3710 | exp_065 |
| 6 | 0.5965 | **0.1213** | 0.3226 | exp_072 |

- Cluster sizes at k=6: `[6, 7, 10, 11, 13, 15]`.

## Reading

Gentle monotone climb (0.093 → 0.116 → 0.121) with falling silhouette and
purity: textbook well-behaved k-curve at negligible absolute levels.
Protocol-insensitive model, mildly k-sensitive — unchanged verdict.
