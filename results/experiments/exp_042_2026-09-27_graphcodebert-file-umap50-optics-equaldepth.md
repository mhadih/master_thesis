# Experiment 042 — GraphCodeBERT whole-file + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_042`
- **Status:** full pipeline — DB-generated whole-file GraphCodeBERT embeddings (see exp_039 §1)
- **Varies vs. exp_016:** embedding protocol only (function-level → whole-file); all else identical

## 1. Setup

Same embeddings (68 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_039. Clustering: **OPTICS** (`cluster_method='xi'`) grid
(min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via
`clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_gbf/op/`. Artifacts: `../models/exp_042_best_clustering.pkl`,
`../figures/exp_042_umap_optics.png`.

## 2. Results

| Metric | exp_042 (whole-file) | exp_016 (function-level) |
|---|---|---|
| Best params | **(ms=9, xi=0.05, inf)** | (ms=8, xi=0.05, inf) |
| Silhouette | 0.5620 | 0.6275 |
| Clusters | **3 + 23 noise** (stored k=3, counting noise); sizes `[10, 14, 15, 23(noise)]` — 37% noise | 3 + 11 noise |
| NMI | 0.0694 | 0.0922 |
| Purity | **0.5000** | 0.5161 |

- Grade-bin borders (3 bins): `[55.39, 95.0, 103.33]` — identical to exp_016 (same k).
- NMI/purity treat noise as an ordinary label; purity 0.50 on 3 bins is a coarseness
  artifact (cf. NMI ≈ 0.07), same as exp_016.

## 3. Component observations (for the conclusion)

- Same operating point (ms≈9, xi=0.05), same verdict (NMI 0.07 vs. 0.09): OPTICS
  agrees with the other three algorithms that whole-file space carries no more
  grade signal than function space — unanimous protocol-insensitivity for
  GraphCodeBERT, opposite to CodeBERT's density-method swings.
- Noise fringe doubles (11 → 23): whole-file groups are looser at the margins.

## 4.–5. Blocked steps / reproduce

As exp_039 (replace `km/kmeans_gbf.py` with `op/optics_gbf.py`).
