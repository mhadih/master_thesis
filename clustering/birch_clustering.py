from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap
import numpy as np
import json
import matplotlib.pyplot as plt
import csv
import pickle

NUMBER_OF_COMPONENTS = 50 # dimention of clustering input
embeddings_filename = "user_CodeT5_embeddings.jsonl"
grades_filename = "user_grades.csv"
clustering_algorithm = "Birch"
output_figure_filename = "umap_with_birch_algorithm.png"

user_embeddings = {}
with open(embeddings_filename, 'r') as f:
    for line in f:
        entry = json.loads(line)
        user_embeddings[entry["user_id"]] = np.array(entry["embedding"], dtype=np.float32)

# Collect all embeddings into one NumPy array
user_ids = list(user_embeddings.keys())

# Read the CSV into a dictionary: {user_id: grade}
grades_dict = {}
with open(grades_filename, mode='r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        uid = int(row[0])
        grade = float(row[1])
        grades_dict[uid] = grade

# --- Assign grades for user_ids ---
ground_truth_labels = []
embeddings = []
for uid in user_ids:
    uid = int(uid)
    if uid in grades_dict:
        grade = grades_dict[uid]
        ground_truth_labels.append(grade)
        embeddings.append(user_embeddings[str(uid)])
    else:
        print(f"Warning: user_id {uid} not found in CSV.")

all_embeddings = np.array(embeddings)

# Step 1: Reduce dimensions for clustering (to 50D for better structure)
umap_reducer = umap.UMAP(n_components=NUMBER_OF_COMPONENTS, random_state=42, metric="cosine")
emb_umap = umap_reducer.fit_transform(all_embeddings)
emb_umap = normalize(emb_umap)  # L2-normalize, same input space as kmeans_clustering.py
    


def find_best_birch_params(emb, threshold_values=None, n_clusters_values=None):
    if threshold_values is None:
        # Thresholds must fit the data scale. On the L2-normalized UMAP-50
        # space pairwise distances are ~0.01-0.07: threshold 0.005 yields ~54
        # subclusters, 0.03 yields ~2 (n=62). Interval [0.005, 0.025], step 0.0025.
        threshold_values = np.arange(0.005, 0.0275, 0.0025)
    if n_clusters_values is None:
        # Try letting Birch decide, plus fixed numbers
        n_clusters_values = [4, 5, 6, 8, 10]

    best_score = -1
    best_params = None
    best_labels = None
    best_n_clusters = None

    for threshold in threshold_values:
        for n_clusters in n_clusters_values:
            birch = Birch(threshold=threshold, n_clusters=n_clusters)
            labels = birch.fit_predict(emb)

            unique_labels = set(labels)
            if len(unique_labels) < 4:
                continue

            score = silhouette_score(emb, labels, metric="cosine")
            print(f"threshold={threshold:.3f}, n_clusters={n_clusters}, silhouette_score={score:.4f}")

            if score > best_score:
                best_score = score
                best_params = (threshold, n_clusters)
                best_labels = labels
                best_n_clusters = len(unique_labels)

    return best_params, best_score, best_labels, best_n_clusters



best_params, best_score, best_labels, best_k = find_best_birch_params(emb_umap)
print("Best params (threshold, n_clusters):", best_params, "with silhouette score:", best_score)

if best_labels is None:
    raise RuntimeError(
        "BIRCH grid found no valid clustering (every config gave < 4 clusters). "
        "Widen/extend threshold_values to fit your embedding scale."
    )

# Step 3: Re-project to 2D using UMAP for visualization
umap_2d = umap.UMAP(n_components=2, random_state=42)
emb_umap_2d = umap_2d.fit_transform(all_embeddings)

# Store best clustering
with open("best_clustering.pkl", "wb") as f:
    pickle.dump((best_k, best_labels), f)

# Step 4: Visualize with best Birch labels
plt.figure(figsize=(10, 7))
scatter = plt.scatter(emb_umap_2d[:, 0], emb_umap_2d[:, 1], c=best_labels, cmap='tab20', s=50, edgecolor='k')
for i, label in enumerate(ground_truth_labels):
    plt.text(emb_umap_2d[i, 0], emb_umap_2d[i, 1], label, fontsize=9)
plt.title(f"UMAP + {clustering_algorithm} Clustering (Best k = {best_k})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.colorbar(scatter, label='Cluster ID')
plt.savefig(output_figure_filename)
plt.show()