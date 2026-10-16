"""Fair fixed-k rerun over all CC2Vec aggregation methods (k=4).

Same pipeline for every method: UMAP-50 (cosine, random_state=42) +
L2-normalize, KMeans k=4 (random_state=42), cosine silhouette, equal-depth
grade bins, NMI + AMI + purity. Bin-border/assign logic mirrors
src/evaluation/evaluation.py.

Env vars:
  ABL_K       number of clusters/bins (default: 4)
  ABL_METHODS comma-separated subset of FILES keys (default: all)
  ABL_OUT     output JSON (default: results/experiments/ablation_cc2vec_k4.json)

Writes JSON rows [{method, file, n, silhouette, nmi, ami, purity, sizes, borders}].
"""

import csv
import json
import os

import umap.umap_ as umap
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR = os.path.join(REPO, "student_embeddings")
GRADES = os.path.join(REPO, "gradesheets", "user_grades.csv")
K = int(os.environ.get("ABL_K", "4"))
OUT = os.environ.get(
    "ABL_OUT", os.path.join(REPO, "results", "experiments", "ablation_cc2vec_k4.json"))

FILES = {
    "baseline": "user_CC2Vec_embeddings.jsonl",
    "editsize": "user_CC2Vec_abl_editsize.jsonl",
    "last5": "user_CC2Vec_abl_last5.jsonl",
    "no1line": "user_CC2Vec_abl_no1line.jsonl",
    "decay": "user_CC2Vec_abl_decay.jsonl",
    "lstm_last_s0": "user_CC2Vec_abl_lstm_last_s0.jsonl",
    "lstm_last_s1": "user_CC2Vec_abl_lstm_last_s1.jsonl",
    "lstm_last_s2": "user_CC2Vec_abl_lstm_last_s2.jsonl",
    "lstm_mean_s0": "user_CC2Vec_abl_lstm_mean_s0.jsonl",
    "lstm_mean_s1": "user_CC2Vec_abl_lstm_mean_s1.jsonl",
    "lstm_mean_s2": "user_CC2Vec_abl_lstm_mean_s2.jsonl",
}


def load_grades():
    grades = {}
    with open(GRADES, mode="r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            grades[int(row[0])] = float(row[1])
    return grades


def compute_bin_borders(grades, k):
    """Equal-depth borders from a sorted grade list (cf. evaluation.py)."""
    n = len(grades)
    bin_size = n // k
    borders = []
    for i in range(1, k):
        index = i * bin_size
        if index >= n:
            index = n - 1
        borders.append(grades[index])
    borders.append(grades[-1])
    return borders


def assign_cluster(grade, borders):
    for i, border in enumerate(borders):
        if grade <= border:
            return i
    return len(borders) - 1


def calculate_purity(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_correct = 0
    for cluster in np.unique(y_pred):
        in_cluster = y_true[np.where(y_pred == cluster)]
        if len(in_cluster) == 0:
            continue
        total_correct += np.sum(in_cluster == mode(in_cluster).mode)
    return total_correct / len(y_true)


def run_method(name, emb_file, grades_dict, k):
    user_embeddings = {}
    with open(os.path.join(EMB_DIR, emb_file), "r") as f:
        for line in f:
            entry = json.loads(line)
            user_embeddings[entry["user_id"]] = np.array(entry["embedding"], dtype=np.float32)
    user_ids = list(user_embeddings.keys())
    embeddings = np.array([user_embeddings[u] for u in user_ids if int(u) in grades_dict])
    grades = [grades_dict[int(u)] for u in user_ids if int(u) in grades_dict]

    space = normalize(
        umap.UMAP(n_components=50, random_state=42, metric="cosine").fit_transform(embeddings))
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(space)
    sil = silhouette_score(space, labels, metric="cosine")

    borders = compute_bin_borders(sorted(grades), k)
    gt = [assign_cluster(grades_dict[int(u)], borders)
          for u in user_ids if int(u) in grades_dict]
    return {
        "method": name,
        "file": emb_file,
        "n": len(gt),
        "silhouette": float(sil),
        "nmi": float(normalized_mutual_info_score(gt, labels)),
        "ami": float(adjusted_mutual_info_score(gt, labels)),
        "purity": float(calculate_purity(gt, labels)),
        "sizes": sorted(int((labels == c).sum()) for c in np.unique(labels)),
        "borders": borders,
    }


def main():
    only = os.environ.get("ABL_METHODS", "")
    names = [m.strip() for m in only.split(",") if m.strip()] or list(FILES)
    unknown = [m for m in names if m not in FILES]
    if unknown:
        raise ValueError(f"unknown methods: {unknown} (choose from {sorted(FILES)})")
    grades_dict = load_grades()
    rows = []
    for name in names:
        row = run_method(name, FILES[name], grades_dict, K)
        rows.append(row)
        print("{method}: n={n} sil={silhouette:.4f} NMI={nmi:.4f} "
              "AMI={ami:.4f} pur={purity:.4f} sizes={sizes}".format(**row), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
