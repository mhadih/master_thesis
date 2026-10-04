# Experiment 079 — GraphCodeBERT + UMAP-50 + KMeans (k=7, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_079`
- **Status:** executed — K=7 counterpart of exp_058/065/072
- **Varies vs. exp_072:** k=6 → k=7; all else identical

## Setup

GraphCodeBERT function embeddings (67 × 768-dim), 62 students, UMAP-50 cosine +
L2-normalize, KMeans k=7, cosine silhouette, equal-depth (7 bins).
Sandbox: `/tmp/opencode/s03run_fx7/i79/`.

Artifacts: `../models/exp_079_best_clustering.pkl`, `../figures/exp_079_umap_kmeans_k7.png`.

## Results (k = 4 → 5 → 6 → 7)

| k | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|
| 4 | 0.6540 | 0.0932 | 0.4032 | exp_058 |
| 5 | 0.6065 | 0.1156 | 0.3710 | exp_065 |
| 6 | 0.5965 | 0.1213 | 0.3226 | exp_072 |
| 7 | 0.5232 | **0.2041** | 0.3226 | exp_079 |

- Cluster sizes at k=7: `[5, 6, 7, 10, 10, 11, 13]`.

## Reading

First real step up (0.12 → 0.20) with silhouette falling throughout (0.65 →
0.52): GraphCodeBERT was the flattest curve until k=7, where something in the
7-way split finally catches grade structure. Still the lowest absolute level
among snapshot models — protocol-insensitive, mildly k-sensitive stands, with
this one uptick noted.
