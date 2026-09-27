# Experiment 043 — E5 function-level + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_043`
- **Status:** full pipeline — embeddings generated locally via fixed DB (see §1)
- **Varies vs. exp_005:** embedding protocol only (whole-file → function-level); model identical (multilingual-e5-base)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, final version per (user, file), cpp+hpp — live DB query | `generate_emb/e5_func_inference.py` (new repo script) |
| Embedding model | **multilingual-e5-base** (768-dim): functions regex-extracted from final file, each with `passage:` prefix + masked mean-pool + L2-norm, word-count-weighted mean per user — mirrors `codeBert_inference.py` structure | `student_embeddings/user_e5_func_embeddings.jsonl` (67 users × 768-dim; user `143` yielded no extractable functions) |
| Students evaluated | **62** (5 ungraded excluded: `7, 142, 145, 146, 148`; graded set = standard 62 minus `143`) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=4) | `src/evaluation/evaluation.py` (unchanged) |

Run sandbox: `/tmp/opencode/s03run_ef/km/`. Artifacts: `../models/exp_043_best_clustering.pkl`,
`../figures/exp_043_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_043 (function-level) | exp_005 (whole-file) |
|---|---|---|
| Best k | **4** | 17 |
| Silhouette | 0.5768 | 0.5252 |
| NMI | 0.0818 | 0.5350 |
| Purity | 0.4194 | 0.3387 |

- Cluster sizes: `[12, 13, 16, 21]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **Protocol collapse:** NMI 0.5350 → 0.0818 — function-level E5 loses essentially all
  grade signal while gaining compactness (sil 0.58 vs. 0.53, k 17→4). The biggest
  protocol swing in the thesis, opposite in direction to CodeBERT (where finer
  granularity helped density methods).
- Hypothesis for discussion: E5's passage-training makes whole-file semantics
  coherent; chopping into functions + re-averaging destroys the document-level
  signal its pretraining relies on. Function splitting suits token models with
  local inductive bias (CodeBERT), not retrieval-trained generalists.
- Population note: user 143 absent (no regex-matched functions); n=62 either way.

## 4.–5. Blocked steps / reproduce

DB-backed Steps 0/1 now work locally; Step 3 ran via the new repo script. Steps 2/4
skipped per workflow note. Reproduce: run `generate_emb/e5_func_inference.py`,
then KMeans + eval on `user_e5_func_embeddings.jsonl` (sandbox s03run_ef/km pattern).
