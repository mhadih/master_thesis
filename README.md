# Master Thesis — Student Source Code Evolution with Transformer-Based Models

Analysis of how student source code evolves over an assignment: code snapshots are
pulled from a PostgreSQL recorder database, embedded with transformer-based code
models, and compared/clustered to study the relation between editing behavior and grades.

Two course datasets are covered: **S03** (root pipeline) and **S04** (`S04/`).

---

## 1. Repository layout

| Path | Contents |
|---|---|
| `src/data_collection/` | DB ingestion: `retrieve_data.py`, `convert_studentId_to_userId.py`, `count_records.py`, `manipulate_content.py` |
| `src/distance_analysis/` | Edit-distance & similarity: `max_edit_dist.py`, `mean_edit_distance.py`, `calculate_distance.py` |
| `generate_emb/` | Embedding inference per model: CodeBERT, CodeT5 (×2 + parser variant), E5, Jina, Qwen |
| `my_ccbert/` | CCBert model code (`uer/` library) + `ccbert_inference.py` |
| `clustering/` | Clustering: KMeans, DBSCAN, BIRCH, Agglomerative, Girvan–Newman, OPTICS |
| `src/visualization/` | `plot_embeddings.py` (PCA / t-SNE / UMAP) |
| `src/evaluation/` | `evaluation.py` (NMI + Purity against grade bins) |
| `src/db_config.py` | Shared DB-config helper (reads `.env`) |
| `S04/` | S04-semester pipeline (dump, edit distance, grading) + its CSVs |
| `data/` | Small derived CSVs (`output-*.csv`, `max_edit_distance.csv`, `student_distance_matrix.csv`) |
| `student_embeddings/` | Per-student embedding JSONL files (one line per `user_id`) |
| `gradesheets/` | Grade CSVs (`user_grades.csv` = `user_id,grades`) |
| `results/figures/` | All PNG plots |
| `results/models/` | `best_clustering.pkl` = `(best_k, best_labels)` |
| `samples/cpp/` | Sample C++ submissions (unrelated to the pipeline) |
| `.env` / `.env.example` | DB credentials (untracked) / template (tracked) |

Large artifacts (`venv/`, `*.bin*`, `my_ccbert/cpp_traces.csv`, pretraining corpus) are
git-ignored on purpose — see `.gitignore`.

---

## 2. Prerequisites

- **Python 3.12** (repo was built with 3.12.3)
- **PostgreSQL** with the recorder databases reachable:
  - S03: `code_recorder` (tables `codetrace`, `users`)
  - S04: `code_recorder_s04` (tables `"CodeTrace"`, `"UserAttributes"`), plus remote LMS access for `dump_db.py`
- **GPU optional** — inference scripts use CUDA when available, otherwise CPU (slow).
- **libclang** only for `generate_emb/codeT5_inference_with_parser.py`
  (path is hardcoded in that file: `/home/hadi/libclang-minimal/lib/libclang.so.16.0.4`).

---

## 3. Installation

```bash
git clone git@github.com:mhadih/master_thesis.git
cd master_thesis

python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tokenizers huggingface-hub safetensors
pip install pandas numpy scikit-learn scipy matplotlib seaborn
pip install umap-learn pynndescent numba tqdm termcolor
pip install psycopg2-binary python-Levenshtein
```

> `python-Levenshtein` and `seaborn` are required by the edit-distance and
> distance-matrix scripts but are easy to miss — make sure both are installed.

### Database credentials

```bash
cp .env.example .env   # then fill in passwords
```

`.env` keys:

| Keys | Used by |
|---|---|
| `CODE_RECORDER_*` (`DBNAME/USER/PASSWORD/HOST/PORT`) | `src/data_collection/*`, `src/distance_analysis/max_edit_dist.py`, `generate_emb/*`, `my_ccbert/ccbert_inference.py` |
| `S04_*` | `S04/edit_distance.py`, `S04/dump_db.py` (local DB) |
| `REMOTE_*` | `S04/dump_db.py` (remote LMS DB) |

`.env` is git-ignored and never committed. Every DB script loads it automatically
via `src/db_config.py` (dependency-free mini loader); real environment variables
take precedence over `.env` values.

---

## 4. Run order (S03 pipeline)

> ⚠️ All scripts use **repository-root-relative paths** — always run from the repo root:
> `python src/...`, `python clustering/...`, etc. Do **not** `cd` into subfolders
> (one exception: `e5/jina/qwen` inference writes to `../student_embeddings/`,
> i.e. they assume CWD = `generate_emb/` — run those three from inside `generate_emb/`).

### Step 0 — Pull raw traces from the DB

```bash
python src/data_collection/retrieve_data.py
```

- Set `assignment_phase = 1/2/3` at the top of the file first.
- Output: `data/output.csv` (phase 1), `output-phase2.csv`, `output-phase3.csv`.

### Step 1 — Map student IDs to DB user IDs (grades)

```bash
python src/data_collection/convert_studentId_to_userId.py
```

- Reads `gradesheets/APS03_CA6_grades.csv` (+ DB `users`) → writes `gradesheets/user_grades.csv`.
- `python src/data_collection/count_records.py` is an optional sanity check (prints per-student record counts).

### Step 2 — Edit-distance features

```bash
python src/distance_analysis/max_edit_dist.py
python src/distance_analysis/mean_edit_distance.py
```

- `max_edit_dist.py` computes the max consecutive Levenshtein distance per
  `(user_id, filename)` from `codetrace WHERE identifier='Text Change'`.
- ⚠️ Known bug: it saves with `result_df.to_csv("max_edit_distance.py")` — rename the
  output to `data/max_edit_distance.csv` (or fix the extension in the file).
- `mean_edit_distance.py` reads `data/max_edit_distance.csv` and prints the mean.

### Step 3 — Generate embeddings (one script per model)

```bash
cd generate_emb
python codeBert_inference.py          # CodeBERT
python codeT5_inference.py            # CodeT5 (+LSTM variant)
python codeT5_inference_with_parser.py  # CodeT5 + clang parser (needs libclang)
python e5_inference.py                # E5  → ../student_embeddings/user_e5_embeddings.jsonl
python jina_inference.py              # Jina → ../student_embeddings/user_jina_embeddings.jsonl
python qwen3_inference.py             # Qwen → ../student_embeddings/user_qwen_embeddings.jsonl
cd ..
python my_ccbert/ccbert_inference.py  # CCBert (needs *.bin-avg weights, git-ignored)
```

- Each script reads `codetrace` from the DB and writes one JSONL line per user:
  `{"user_id": ..., "embedding": [...]}` into `student_embeddings/`.
- Download the CCBert weights (`ccbert_small.bin-avg`) separately — they are too
  large for git. `my_ccbert/cpp_traces.csv` (1.3 GB, git-ignored) is produced here too.

### Step 4 — Distance matrix

```bash
python src/distance_analysis/calculate_distance.py
```

- Reads the embeddings JSONL (filename hardcoded at the top of the file) +
  `gradesheets/user_grades.csv` → cosine-similarity → normalized distance matrix.
- Output: `data/student_distance_matrix.csv`, `results/figures/student_distance_matrix.png`.

### Step 5 — Clustering (pick one)

```bash
python clustering/kmeans_clustering.py
python clustering/dbscan_clustering.py
python clustering/birch_clustering.py
python clustering/agglomerative_clustering.py
python clustering/girvan_clustering.py
python clustering/optics_clustering.py
```

- Each script hardcodes its input embedding file and output PNG at the top —
  edit `embeddings_filename` / `output_figure_filename` to switch models.
- Each writes `results/models/best_clustering.pkl` (`(best_k, best_labels)`) and a
  `UMAP + <algo>` scatter plot annotated with grades into `results/figures/`.
- ⚠️ They all overwrite the same `best_clustering.pkl` — rename it after each run
  if you want to keep several results.

### Step 6 — Embedding plots (optional)

```bash
python src/visualization/plot_embeddings.py
```

- PCA / t-SNE / UMAP 2-D scatters → `results/figures/{pca,tsne,umap}_plot.png`.
- (Input file is hardcoded at the top of the script.)
- Note: `src/visualization/visualization.py` is a corrupted placeholder (null bytes), ignore it.

### Step 7 — Evaluate clustering vs. grades

```bash
python src/evaluation/evaluation.py
```

- Reads `results/models/best_clustering.pkl` + embeddings (to fix student order) +
  grades; builds equal-depth grade bins as ground truth.
- Prints **NMI** and **Purity**. Input filenames are hardcoded at the top.

---

## 5. Run order (S04 pipeline)

All files live in `S04/` — run from the repo root so relative CSV paths resolve.

```bash
python S04/dump_db.py                  # 1. copy UserAttributes from remote LMS → local S04 DB
python S04/edit_distance.py            # 2. mean max-edit-distance per student → student_mean_edit_distance_phase3.csv
python S04/assign_grade.py             # 3. activity-count scores: phase3.csv → phase3_with_scores.csv
python S04/assign_edit_distance_grade.py  # 4. qcut edit-distance scores → edit_distance_grades_phase3.csv
```

- `edit_distance.py` targets the Phase-3 assignment (see the `WHERE` clause) and
  skips the `blocked_student_id` list — adjust both for other phases.

---

## 6. Reproducing the committed results

The checked-in artifacts correspond to: CodeT5-file embeddings →
`calculate_distance.py` → OPTICS/KMeans/BIRCH/DBSCAN clustering → `evaluation.py`
with Jina embeddings. To reproduce end-to-end, follow S03 Steps 0, 1, 3 (CodeT5),
4, 5, 7 in order.

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: user_*.jsonl / user_grades.csv` | Wrong CWD — run from repo root (see §4 note) |
| `ModuleNotFoundError: Levenshtein / seaborn` | Missing packages — `pip install python-Levenshtein seaborn` |
| `psycopg2.OperationalError` / auth failed | `.env` missing or wrong — check §3 |
| `sts ... .bin-avg` / model weights missing | Git-ignored weights — obtain separately |
| `libclang` error in parser variant | Fix the hardcoded libclang path in the script |
| Empty `user_qwen_embeddings.jsonl` | Qwen inference never completed in this checkout — rerun it |
| `best_clustering.pkl` doesn't match your run | A later clustering script overwrote it — rerun the one you want |
