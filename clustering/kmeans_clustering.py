from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import umap.umap_ as umap
import numpy as np
import json
import matplotlib.pyplot as plt
import csv
import pickle

NUMBER_OF_COMPONENTS = 50 # dimention of clustering input
embeddings_filename = "./student_embeddings/user_e5_embeddings.jsonl"
grades_filename = "./gradesheets/user_grades.csv"
output_figure_filename = "umap_with_grades.png"

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

# Step 1: Reduce dimensions for clustering
umap_reducer = umap.UMAP(n_components=NUMBER_OF_COMPONENTS, random_state=42, metric="cosine")
emb_umap_10d = umap_reducer.fit_transform(all_embeddings)

embeddings = normalize(emb_umap_10d)

# Step 2: Try KMeans for multiple values of k and evaluate silhouette scores
k_values = [i for i in range(4, 21)]
best_k = None
best_score = -1
best_labels = None

print("Evaluating KMeans clustering with different k values:")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  k = {k} → Silhouette Score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels


print(f"\n✅ Best k found: {best_k} with Silhouette Score = {best_score:.4f}")

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
plt.title(f"UMAP + KMeans Clustering (Best k = {best_k})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.colorbar(scatter, label='Cluster ID')
plt.savefig(output_figure_filename)
plt.show()