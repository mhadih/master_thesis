# Experiment 089 — CodeBERT whole-file + PCA-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-10-07
- **ID:** `exp_089`
- **Status:** executed — reducer variant (see reducer_comparison_dim50.md)
- **Varies vs. exp_038:** reducer only (UMAP-50 → PCA-50, deterministic); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CodeBERT whole-file** (68 × 768-dim), 62 students | `student_embeddings/user_CodeBert_file_embeddings.jsonl` |
| Reduction | **PCA-50** (`random_state=42`), then L2-normalize | `/tmp/opencode/s03run_red/reducer_sweep.py` |
| Clustering | **OPTICS** `xi` grid, best cosine silhouette subject to ≥4 non-noise clusters | same as exp_038 |
| Eval | equal-depth, bins = winner's k (=4) | same script family |

Artifacts: `../models/exp_089_best_clustering.pkl`.

## 2. Results

| Metric | exp_089 (PCA) | exp_038 (UMAP) |
|---|---|---|
| Best params | **(ms=3, xi=0.01, inf)** | (ms=2, xi=0.01, inf) |
| Silhouette | 0.1305 | 0.3903 |
| Clusters | **4 + 26 noise**; sizes `[3, 4, 10, 19]`, noise 26 (42%) | 20 + 6 noise |
| NMI (4 bins) | **0.1527** | 0.6641 |
| Purity | 0.4032 | 0.4194 |

## 3. Component observations (for the conclusion)

- Linear PCA keeps global variance but destroys the local neighborhood structure
  OPTICS needs: few compact groups, massive noise, weak NMI. The linear baseline
  every nonlinear reducer must beat — UMAP beats it 4× (0.66 vs. 0.15).
- Deterministic (no seed variance to report) — the only such reducer in the sweep.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_red/reducer_sweep.py  # PCA section
```
