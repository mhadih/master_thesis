# Experiment 017 — CCBERT + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_017`
- **Status:** full pipeline — embeddings freshly generated (see §1); DB-backed repo scripts still blocked
- **Varies vs. exp_001/005/009/013:** embedding model only (→ CCBERT, change-sequence input); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, consecutive per-(user, file) change pairs, cpp files | `my_ccbert/cpp_traces.csv` (DB down, so offline CSV used) |
| Embedding model | **CCBERT-small** (uer transformer, hidden 512, 4 layers, local `ccbert_small.bin-avg`): difflib-aligned old/new/edit token ids per change → encoder(old_ids) mean-pooled per change → mean per file → token-length-weighted average per user — mirrors `my_ccbert/ccbert_inference.py` | 68 CodeT5 users → `student_embeddings/user_CCBERT_file_embeddings.jsonl` (66 users × 512-dim; 2 targets had no usable pairs; ~120.7k pairs, ~3.4 h on MX150, batched 32) |
| Students evaluated | **62** (4 ungraded excluded: `142, 143, 145, 146` — identical graded set to exp_001) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=5) | `src/evaluation/evaluation.py` (unchanged) |

Notes: (a) repo script contains a leftover test `break` after the first user (hence the
old 1-line artifact) — sandbox runner omits it. (b) Like the repo script, only
`old_ids` feed the encoder; `new_ids`/`edit_ids` are computed but unused. (c) Tokenizer
is the HF `microsoft/codebert-base` tokenizer (small download, cached afterwards).

Run sandbox: `/tmp/opencode/s03run_ccb/km/`. Artifacts: `../models/exp_017_best_clustering.pkl`,
`../figures/exp_017_umap_kmeans_k5.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **5** |
| **Silhouette (cosine)** | **0.7030** |
| **NMI** (vs. 5 equal-depth grade bins) | **0.1208** |
| **Purity** | **0.4032** |

- Sweep: k=4: 0.6913, k=5: **0.7030** ✅, 6: 0.6588, 7: 0.6019, 8–13: 0.46–0.53, 14–20: 0.38–0.47.
- Cluster sizes: `[8, 10, 13, 15, 16]`.
- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **Change-sequence vs. token-sequence models:** CCBERT/KMeans (sil 0.7030, NMI 0.1208)
  lands squarely in the CodeBERT/GraphCodeBERT band (sil 0.65–0.70, NMI ≈ 0.09–0.10),
  far from CodeT5/e5 (NMI ≈ 0.54). Modeling *edits* instead of *code* did not buy
  grade relevance here.
- Purity 0.4032 on 5 bins matches the flat cross-series band (0.29–0.52, mostly
  bin-count artifacts).

## 4. Blocked steps

Repo DB scripts (Steps 0/1/3) still blocked; embeddings generated from the offline
CSV instead. Steps 2/4 skipped per workflow note.

## 5. Reproduce

```bash
# 1. embeddings (needs my_ccbert weights + codebert tokenizer in HF cache; CUDA optional)
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_ccb/ccb_from_csv.py
# 2. clustering
mkdir -p /tmp/opencode/s03run_ccb/km && cd /tmp/opencode/s03run_ccb/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_ccb.py / evaluate_ccb.py = repo scripts with embeddings → .../user_CCBERT_file_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_ccb.py
/home/hadi/thesis/venv/bin/python evaluate_ccb.py
```
