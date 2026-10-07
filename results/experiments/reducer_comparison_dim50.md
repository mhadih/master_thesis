# Reducer comparison at dim=50 — CodeBERT whole-file + OPTICS (S03)

- **Date (UTC):** 2026-10-07
- **IDs:** exp_088 (t-SNE), exp_089 (PCA), exp_090–092 (Autoencoder seeds 0–2);
  UMAP-50 reference = exp_038 (not rerun)
- **Status:** executed — dimensionality-reduction method varied, all else fixed
- **Varies:** reducer only (LDA excluded by design, see §4)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CodeBERT whole-file** (68 × 768-dim), 62 students | `student_embeddings/user_CodeBert_file_embeddings.jsonl` |
| Reduction | dim=50: **t-SNE** (exact, cosine, perplexity 30), **PCA**, **Autoencoder** (768-256-64-50 MLP, MSE, full-batch 200 epochs, seeds 0–2, train MSE ≈ 0.00184 all seeds); **UMAP-50** reference from exp_038 | `/tmp/opencode/s03run_red/reducer_sweep.py` |
| Post | L2-normalize (all conditions) | — |
| Clustering | **OPTICS** `xi` grid (ms 2…9 × xi [0.01, 0.05, 0.1, 0.2]), best cosine silhouette subject to ≥4 non-noise clusters | same as exp_038 |
| Eval | equal-depth, bins = winner's k | same script family |

Artifacts: `../models/exp_08{8,9}_best_clustering.pkl`,
`../models/exp_09{0,1,2}_best_clustering.pkl` (AE seeds).

## 2. Results

| Reducer | Exp | Winner | Silhouette | k (+noise) | NMI | Purity |
|---|---|---|---|---|---|---|
| UMAP-50 (ref) | 038 | (2, 0.01) | 0.3903 | 20+6n | **0.6641** | 0.4194 |
| t-SNE-50 | 088 | (2, 0.01) | −0.0192 | 8+31n [2,2,2,3,4,4,5,9] | 0.2934 | 0.3226 |
| PCA-50 | 089 | (3, 0.01) | 0.1305 | 4+26n [3,4,10,19] | 0.1527 | 0.4032 |
| AE-50 s0 | 090 | (2, 0.01) | −0.0781 | 14+29n [2×10,3×3,4] | 0.4448 | 0.3387 |
| AE-50 s1 | 091 | (4, 0.01) | −0.0684 | 4+30n [7,7,9,9] | 0.1609 | 0.4355 |
| AE-50 s2 | 092 | (3, 0.01) | −0.0644 | 6+29n [3,3,5,5,5,12] | 0.2564 | 0.4032 |
| AE mean±std | — | — | −0.070±0.006 | — | 0.287±0.117 | 0.393±0.040 |

## 3. Component observations (for the conclusion)

- **UMAP wins outright** (0.66 vs. next-best AE seed 0.44): the reducer is as
  decisive as the embedding model — same data, same clusterer, 2× NMI gap
  between best and runner-up reducers.
- **Negative-silhouette spaces still score** (t-SNE −0.02 → NMI 0.29; AE ≈ −0.07
  → NMI up to 0.44): without UMAP's clustering-friendly geometry, OPTICS finds
  noise-dominated micro-partitions (~half the students labeled noise in all
  four non-UMAP runs) that nonetheless overlap grades well above chance. Same
  compactness-≠-relevance moral, now with the sign flipped.
- **AE is seed-unstable** (NMI 0.16–0.44, k = 4/6/14 across seeds at identical
  train loss): the latent geometry that OPTICS sees varies wildly with init —
  report mean±std (0.287±0.117), never a single seed. A VAE or fixed-projection
  control would be needed before trusting any single AE number.
- **t-SNE at 50 dims is off-label use** (it is designed for 2–3D visualization);
  included for completeness at the user's request — its 0.29 beats PCA/AE-mean
  anyway, which says more about those baselines than about t-SNE.

## 4. Why LDA is excluded (documented, not omitted)

LDA is supervised (needs labels — incompatible with the unsupervised protocol)
and capped at `classes − 1` dims: dim=50 would require ≥51 grade classes from
62 students (≈1 student per class — degenerate). No valid LDA configuration
exists at this operating point.

## 5. Reproduce

```bash
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_red/reducer_sweep.py
# CPU, ~15 min (AE training dominates); needs 62-student CodeBERT-file embeddings
```
