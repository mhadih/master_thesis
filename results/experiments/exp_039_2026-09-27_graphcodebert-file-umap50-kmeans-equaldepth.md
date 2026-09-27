# Experiment 039 — GraphCodeBERT whole-file + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_039`
- **Status:** full pipeline — embeddings generated locally via fixed DB (see §1)
- **Varies vs. exp_013:** embedding protocol only (function-level + word-weighted → whole-file [CLS]); model identical (GraphCodeBERT)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, final version per (user, file), cpp+hpp — live DB query (DB connection fixed this session) | `generate_emb/graphCodeBert_file_inference.py` (new repo script) |
| Embedding model | **GraphCodeBERT** (`microsoft/graphcodebert-base`, 768-dim): entire final file in one forward pass, **[CLS] readout**, word-count-weighted mean per user | `student_embeddings/user_GraphCodeBert_file_embeddings.jsonl` (68 users × 768-dim; identical user set to CodeT5) |
| Students evaluated | **62** (6 ungraded excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=4) | `src/evaluation/evaluation.py` (unchanged) |

Run sandbox: `/tmp/opencode/s03run_gbf/km/`. Artifacts: `../models/exp_039_best_clustering.pkl`,
`../figures/exp_039_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_039 (whole-file) | exp_013 (function-level) |
|---|---|---|
| Best k | **4** | 4 |
| Silhouette | 0.5966 | 0.6540 |
| NMI | 0.1041 | 0.0932 |
| Purity | 0.4355 | 0.4032 |

- Cluster sizes: `[10, 15, 18, 19]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- Protocol effect ≈ nil (NMI 0.104 vs. 0.093): whole-file vs. function-level changes
  nothing under KMeans — GraphCodeBERT is insensitive to this protocol choice,
  unlike CodeBERT where density methods swung wildly.
- Same k=4 on both protocols; purity uptick is noise-level.

## 4.–5. Blocked steps / reproduce

DB-backed Steps 0/1 now work locally (connection fixed); Step 3 ran via the new
repo script. Steps 2/4 skipped per workflow note. Reproduce: run
`generate_emb/graphCodeBert_file_inference.py`, then KMeans + eval on
`user_GraphCodeBert_file_embeddings.jsonl` (sandbox s03run_gbf/km pattern).
