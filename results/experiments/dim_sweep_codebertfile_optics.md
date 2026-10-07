# Dimension-reduction sweep — CodeBERT whole-file + OPTICS (S03)

- **Date (UTC):** 2026-10-07
- **IDs:** exp_083 (d=5), exp_084 (d=10), exp_085 (d=25), exp_086 (d=50), exp_087 (raw 768)
- **Status:** executed — UMAP dimensionality varied on the max-NMI setup (exp_038)
- **Varies vs. exp_038:** UMAP n_components only; model, cosine metric, L2-normalize,
  OPTICS xi grid, equal-depth eval identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CodeBERT whole-file** (68 × 768-dim), 62 students | `student_embeddings/user_CodeBert_file_embeddings.jsonl` |
| Reduction | **UMAP** d ∈ {5, 10, 25, 50} (cosine, `random_state=42`) + **raw-768** (L2-normalized, no reduction — UMAP cannot embed 62 samples above ~61 dims, so the top point skips UMAP by design) | `/tmp/opencode/s03run_dim/dim_sweep.py` |
| Clustering | **OPTICS** (`cluster_method='xi'`) grid: ms 2…9 × xi [0.01, 0.05, 0.1, 0.2], best cosine silhouette subject to ≥4 non-noise clusters | same as exp_038 |
| Eval | equal-depth, bins = winner's k | same script family |

Artifacts: `../models/exp_08{3,4,5,6,7}_best_clustering.pkl`.

## 2. Results

| Dim | Exp | Winner (ms, xi) | Silhouette | k (+noise) | NMI | Purity |
|---|---|---|---|---|---|---|
| 5 | 083 | (7, 0.01) | 0.3625 | 4 | 0.1276 | 0.4355 |
| 10 | 084 | (2, 0.01) | 0.4581 | 17 | 0.5657 | 0.4032 |
| 25 | 085 | (7, 0.01) | 0.3703 | 4 | 0.1298 | 0.4194 |
| 50 | 086 | (2, 0.01) | 0.3903 | 20 | 0.6641 | 0.4194 |
| 768 (raw) | 087 | (2, 0.01) | **−0.1169** | 10 | 0.3183 | 0.2742 |

- exp_086 reproduces exp_038 exactly (same params, sil, k, NMI, purity) —
  determinism control for the sweep harness.
- Grade bins follow k (4/17/4/20/10); same equal-depth rule throughout.

## 3. Component observations (for the conclusion)

- **Reduction is load-bearing, non-monotonically:** NMI goes 0.13 (d=5) → 0.57
  (d=10) → 0.13 (d=25) → 0.66 (d=50) → 0.32 (raw). Both extremes (5-dim squeeze,
  raw 768-dim) destroy fine grade structure; the two peaks (10, 50) are both
  high-k solutions (17, 20 clusters) — dimensionality and cluster-count interact:
  only roomy-enough spaces let OPTICS find the fine partitions grades reward.
- **Raw space has *negative* silhouette (−0.12) yet NMI 0.32:** without reduction
  there are no compact groups at all, but a 10-way cut still beats chance —
  further evidence that compactness ≠ grade relevance, now from the reduction
  axis instead of the model axis.
- **d=50 is validated, not arbitrary:** it is the only tested dim reaching the
  series-max NMI, which retroactively justifies the UMAP-50 constant used in
  exp_001–082 — add that sentence to the methods section.
- Caveat: dims between 50 and 768 are untestable with UMAP on n=62 (spectral
  layout needs components < samples); the gap is structural, not an omission.
  A PCA-bridged variant could fill it if a reviewer asks.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_dim/dim_sweep.py
# (sandbox script; inputs/outputs documented in §1; ~10 min, CPU)
```
