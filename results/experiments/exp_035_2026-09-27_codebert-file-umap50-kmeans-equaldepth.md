# Experiment 035 — CodeBERT whole-file + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_035`
- **Status:** partial — embeddings generated on Kaggle (see §1); clustering/evaluation executed locally; DB-backed repo scripts still blocked
- **Varies vs. exp_009:** embedding protocol only (function-level + word-weighted → whole-file [CLS]); model identical (CodeBERT)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, final version per (user, file), cpp+hpp | `my_ccbert/cpp_traces.csv` via Drive → Kaggle (DB down) |
| Embedding model | **CodeBERT** (`microsoft/codebert-base`, 768-dim): entire final file in one forward pass, **[CLS] readout**, word-count-weighted mean per user — mirrors `generate_emb/codebert_file_inference_csv.py` | Kaggle GPU run → `student_embeddings/user_CodeBert_file_embeddings.jsonl` (68 users × 768-dim; identical user set to CodeT5) |
| Students evaluated | **62** (6 ungraded excluded: `7, 142, 143, 145, 146, 148`) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=6) | `src/evaluation/evaluation.py` (unchanged) |

Run sandbox: `/tmp/opencode/s03run_cbf/km/`. Artifacts: `../models/exp_035_best_clustering.pkl`,
`../figures/exp_035_umap_kmeans_k6.png`.

## 2. Results

| Metric | exp_035 (whole-file) | exp_009 (function-level) |
|---|---|---|
| Best k | **6** | 4 |
| Silhouette | 0.4767 | 0.7016 |
| NMI | 0.1105 | 0.1036 |
| Purity | 0.2903 | 0.4032 |

- Cluster sizes: `[7, 7, 10, 11, 12, 15]`.
- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]`.

## 3. Component observations (for the conclusion)

- **Whole-file vs. function-level is nearly a wash on NMI** (0.1105 vs. 0.1036):
  finer function granularity buys compactness (sil 0.70 vs. 0.48) but no extra grade
  signal — consistent with the compactness-≠-relevance pattern.
- Purity drop (0.40 → 0.29) is mostly the 4→6 bin effect, not a real regression.

## 4. Blocked steps

Repo DB scripts (Steps 0/1/3) still blocked; embeddings generated off-DB (Kaggle).
Steps 2/4 skipped per workflow note.

## 5. Reproduce

```bash
# 1. embeddings on Kaggle/Colab GPU: CODEBERT_BATCH=64 python generate_emb/codebert_file_inference_csv.py
# 2. clustering locally
mkdir -p /tmp/opencode/s03run_cbf/km && cd /tmp/opencode/s03run_cbf/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_cbf.py / evaluate_cbf.py = repo scripts with embeddings → .../user_CodeBert_file_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_cbf.py
/home/hadi/thesis/venv/bin/python evaluate_cbf.py
```
