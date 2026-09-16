# Experiment 013 — GraphCodeBERT + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_013`
- **Status:** full pipeline — embeddings freshly generated (see §1); DB-backed repo scripts still blocked
- **Varies vs. exp_001/005/009:** embedding model only (→ GraphCodeBERT); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, final version per (user, file), cpp files | `my_ccbert/cpp_traces.csv` (132k snapshots; DB down, so offline CSV used — same content the DB query would return, minus hpp files) |
| Embedding model | **GraphCodeBERT** (`microsoft/graphcodebert-base`, 768-dim, [CLS] per function, word-count-weighted average per user) — same protocol as `codeBert_inference.py` | `generate_emb/graphCodeBert_inference.py` (new repo script, DB-based) run via sandbox `gcb_from_csv.py` over 68 CodeT5 users → `student_embeddings/user_GraphCodeBert_embeddings.jsonl` (67 users; user `143` yielded no extractable functions) |
| Students evaluated | **62** (5 ungraded excluded: `7, 142, 145, 146, 148`) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=4) | `src/evaluation/evaluation.py` (unchanged) |

Notes: (a) weights loaded from a safetensors conversion placed in the HF cache —
verified tensor-for-tensor against the official `pytorch_model.bin` (203/203 encoder
tensors identical; only unused `lm_head.decoder.*` absent); scripts pass
`use_safetensors=True` (torch<2.6 guard, CVE-2025-32434). (b) Inference ran on CUDA
(MX150), 3302 function snippets. (c) Population differs from exp_001 by one student
(143 absent) — negligible for comparison.

Run sandbox: `/tmp/opencode/s03run_gcb/km/`. Artifacts: `../models/exp_013_best_clustering.pkl`,
`../figures/exp_013_umap_kmeans_k4.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **4** |
| **Silhouette (cosine)** | **0.6540** |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0932** |
| **Purity** | **0.4032** |

- Sweep: k=4: **0.6540** ✅, 5: 0.6065, 6: 0.5965, 7: 0.5232, 8: 0.5278, 9: 0.5219,
  10: 0.5130, 11: 0.5456, 12: 0.5017, 13: 0.5264, 14–20: 0.34–0.42 (sharp drop).
- Cluster sizes: `[10, 14, 15, 23]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **Embedding (GraphCodeBERT vs. CodeBERT, same protocol):** silhouette 0.6540 vs.
  0.7016, NMI 0.0932 vs. 0.1036 — near-identical behavior: very compact coarse
  partitions with no grade relevance. Data-flow pretraining adds nothing here over
  plain CodeBERT under KMeans.
- Compactness-≠-relevance pattern holds for the 4th embedding model.

## 4. Blocked steps

Repo DB scripts (Steps 0/1/3) still blocked (no `code_recorder` DB); embeddings were
generated from the offline CSV instead. Steps 2/4 skipped per workflow note.

## 5. Reproduce

```bash
# 1. embeddings (needs HF cache safetensors + CUDA/CPU)
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_gcb/gcb_from_csv.py
# 2. clustering
mkdir -p /tmp/opencode/s03run_gcb/km && cd /tmp/opencode/s03run_gcb/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_gcb.py / evaluate_gcb.py = repo scripts with embeddings → .../user_GraphCodeBert_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_gcb.py
/home/hadi/thesis/venv/bin/python evaluate_gcb.py
```
