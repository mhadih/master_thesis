# Experiment 047 — CodeT5 function-level + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_047`
- **Status:** full pipeline — embeddings generated locally via fixed DB (see §1)
- **Varies vs. exp_001:** embedding protocol only (whole-file → function-level); model identical (CodeT5-base, encoder mean-pool)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, final version per (user, file), cpp+hpp — live DB query | `generate_emb/codeT5_func_inference.py` (new repo script) |
| Embedding model | **CodeT5-base** (T5 encoder, 768-dim): functions regex-extracted from final file, encoder mean-pool per function, word-count-weighted mean per user | `student_embeddings/user_CodeT5_func_embeddings.jsonl` (67 users × 768-dim; user `143` yielded no extractable functions; weights downloaded to /tmp, converted to safetensors for torch<2.6) |
| Students evaluated | **62** (5 ungraded excluded; graded set = standard 62 minus `143`) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=5) | `src/evaluation/evaluation.py` (unchanged) |

Run sandbox: `/tmp/opencode/s03run_ct5f/km/`. Artifacts: `../models/exp_047_best_clustering.pkl`,
`../figures/exp_047_umap_kmeans_k5.png`.

## 2. Results

| Metric | exp_047 (function-level) | exp_001 (whole-file) |
|---|---|---|
| Best k | **5** | 17 |
| Silhouette | 0.6743 | 0.5179 |
| NMI | 0.0886 | **0.5356** |
| Purity | 0.2903 | 0.3548 |

- Cluster sizes: `[6, 10, 14, 14, 18]`.
- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **The mirror image of CodeBERT:** where CodeBERT's density methods gained from
  finer granularity, CodeT5 collapses (NMI 0.5356 → 0.0886) — function splitting
  destroys exactly the document-level coherence that CodeT5's identifier-aware
  pretraining exploits, same mechanism hypothesized for E5 (exp_043: 0.535 → 0.08).
- Compactness up (0.67 vs. 0.52), k down (5 vs. 17): coarse tight groups with no
  grade content — compactness-≠-relevance once more.

## 4.–5. Blocked steps / reproduce

DB-backed Steps 0/1 now work locally; Step 3 ran via the new repo script (weights:
HF `Salesforce/codet5-base` .bin → local safetensors conversion for torch<2.6).
Steps 2/4 skipped per workflow note. Reproduce: run
`generate_emb/codeT5_func_inference.py`, then KMeans + eval on
`user_CodeT5_func_embeddings.jsonl` (sandbox s03run_ct5f/km pattern).
