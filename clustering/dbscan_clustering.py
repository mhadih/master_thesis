from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap
import numpy as np
import json
import matplotlib.pyplot as plt
import csv
import pickle

NUMBER_OF_COMPONENTS = 50 # dimention of clustering input
embeddings_filename = "user_CodeT5_file_with_LSTM.jsonl"
grades_filename = "user_grades.csv"
output_figure_filename = "umap_with_dbscan.png"

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

# Step 1: Reduce dimensions for clustering (to 10D for better structure)
umap_reducer = umap.UMAP(n_components=NUMBER_OF_COMPONENTS, random_state=42)
emb_umap = umap_reducer.fit_transform(all_embeddings)

# Step 2: Try DBSCAN for multiple values of eps and min_samples and evaluate silhouette scores
def compute_k_distances(emb, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(emb)
    distances, _ = neighbors_fit.kneighbors(emb)
    k_distances = distances[:, -1]  # distance to k-th nearest neighbor
    return np.sort(k_distances)

def get_candidate_eps_values(k_distances, num_values=20):
    return np.linspace(k_distances[0], k_distances[-1], num=num_values)

def find_best_dbscan_params(emb, min_samples=4, num_eps_values=20):
    k_distances = compute_k_distances(emb, min_samples)
    eps_values = get_candidate_eps_values(k_distances, num_eps_values)
    best_score = -1
    best_params = None
    best_labels = None
    for eps in eps_values:
        for min_samples in range(2, 10):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(emb)
            # skip if all noise or one cluster
            unique_labels = set(labels)
            if len(unique_labels) <= 1 or (set(labels) == {-1}):
                continue
            n_clusters = len(unique_labels)
            if n_clusters < 4:
                continue
            score = silhouette_score(emb, labels)
            print(f"eps={eps:.3f}, min_samples={min_samples}, silhouette_score={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
                best_labels = labels
                best_n_clusters = n_clusters

    return best_params, best_score, best_labels, best_n_clusters

best_params, best_score, best_labels, best_k = find_best_dbscan_params(emb_umap, min_samples=4, num_eps_values=20)
print("Best params (eps, min_samples):", best_params, "with silhouette score:", best_score)

# Step 3: Re-project to 2D using UMAP for visualization
umap_2d = umap.UMAP(n_components=2, random_state=42)
emb_umap_2d = umap_2d.fit_transform(all_embeddings)

# Store best clustering
with open("best_clustering.pkl", "wb") as f:
    pickle.dump((best_k, best_labels), f)

# Step 4: Visualize with best KMeans labels
plt.figure(figsize=(10, 7))
scatter = plt.scatter(emb_umap_2d[:, 0], emb_umap_2d[:, 1], c=best_labels, cmap='tab20', s=50, edgecolor='k')
for i, label in enumerate(ground_truth_labels):
    plt.text(emb_umap_2d[i, 0], emb_umap_2d[i, 1], label, fontsize=9)
plt.title(f"UMAP + DBSCAN Clustering (Best k = {best_k})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.colorbar(scatter, label='Cluster ID')
plt.savefig(output_figure_filename)
plt.show()