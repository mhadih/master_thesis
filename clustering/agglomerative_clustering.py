import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle

distance_filename = "student_distance_matrix.csv"
MIN_CLUSTER = 4
MAX_CLUSTER = 20

# === Load normalized distance matrix ===
df_dist = pd.read_csv(distance_filename, index_col=0)
distance_matrix = df_dist.values

# Make sure diagonal is zero
np.fill_diagonal(distance_matrix, 0)

students = df_dist.index.tolist()

best_k = None
best_score = -1
best_labels = None

for k in range(MIN_CLUSTER, MAX_CLUSTER + 1):
    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',  # Use the distance matrix directly
        linkage='average'  
    )
    labels = clustering.fit_predict(distance_matrix)

    # Calculate silhouette score with distance matrix
    score = silhouette_score(distance_matrix, labels, metric='precomputed')
    print(f"k={k}, silhouette score={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"\nBest k: {best_k} with silhouette score: {best_score:.4f}")

# Optional: Print cluster assignment for each student
for student, label in zip(students, best_labels):
    print(f"{student}: Cluster {label}")

# Store best clustering
with open("best_clustering.pkl", "wb") as f:
    pickle.dump((best_k, best_labels), f)
