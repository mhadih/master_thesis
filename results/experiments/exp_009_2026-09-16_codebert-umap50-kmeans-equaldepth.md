# Experiment 009 — CodeBERT + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_009`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_001/exp_005:** embedding model only (CodeT5/e5 → CodeBERT); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeBERT** (`microsoft/codebert-base`, 768-dim, [CLS], function-level + word-count-weighted per user) | `student_embeddings/user_CodeBert_embeddings.jsonl` (67 users; 5 ungraded: `7, 142, 145, 146, 148`) |
| Students evaluated | **62** | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=4) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Note: `generate_emb/codeBert_inference.py` itself needs the DB (blocked) and writes a
different filename (`user_CodeBert_LSTM_embeddings.jsonl`); the evaluated artifact is
the pre-existing `user_CodeBert_embeddings.jsonl`.

Run sandbox: `/tmp/opencode/s03run_cb/km/`. Artifacts: `../models/exp_009_best_clustering.pkl`,
`../figures/exp_009_umap_kmeans_k4.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **4** |
| **Silhouette (cosine)** | **0.7016** |
| **NMI** (vs. 4 equal-depth grade bins) | **0.1036** |
| **Purity** | **0.4032** |

- Full sweep: k=4: **0.7016** ✅, 5: 0.5889, 6: 0.4692, 7: 0.5041, 8: 0.6033, 9: 0.5485,
  10: 0.5088, 11: 0.4404, 12: 0.4610, 13: 0.4455, 14: 0.4721, 15: 0.4954, 16: 0.5001,
  17: 0.5357, 18: 0.5622, 19: 0.5560, 20: 0.5167.
- Cluster sizes at k=4: `[7, 14, 20, 21]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **Embedding (CodeBERT vs. CodeT5/e5):** silhouette 0.7016 is the best of all 12
  experiments by far — CodeBERT space is highly compact under KMeans — yet NMI
  0.1036 is the worst: compactness here does NOT track grades. Sharpest evidence yet
  for the compactness-≠-relevance thesis point.
- Purity 0.4032 is inflated by the coarse 4-bin setup (same confound as ever);
  fixed-bin comparison still open.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_cb/km && cd /tmp/opencode/s03run_cb/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_cb.py = clustering/kmeans_clustering.py with embeddings → .../user_CodeBert_embeddings.jsonl
# evaluate_cb.py = src/evaluation/evaluation.py with embeddings → .../user_CodeBert_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_cb.py  # → best k, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_cb.py               # → NMI, purity
```
