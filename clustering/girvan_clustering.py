import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
from sklearn.metrics import silhouette_score
import numpy as np


distance_filename = "student_distance_matrix.csv"
MIN_CLUSTER = 4
MAX_CLUSTER = 20

# === Step 1: Load distance matrix ===
df_dist = pd.read_csv(distance_filename, index_col=0)

# === Step 2: Build weighted graph from distance matrix ===
G = nx.Graph()
students = df_dist.index.tolist()

for i in range(len(students)):
    for j in range(i+1, len(students)):
        G.add_edge(students[i], students[j], weight=df_dist.iloc[i, j])

# === Step 3: Girvan–Newman clustering & silhouette scoring ===
best_k = None
best_score = -1
best_partition = None

# Girvan–Newman generates hierarchical splits
comp_gen = girvan_newman(G)

# Convert distance to similarity for silhouette (since silhouette expects a "distance metric" if given precomputed distances)
distance_matrix = df_dist.values
np.fill_diagonal(distance_matrix, 0)  # zero diagonal for silhouette_score

for communities in comp_gen:
    # Convert tuple of sets to list of labels
    communities = tuple(sorted(c) for c in communities)
    k = len(communities)

    # Assign each student a label
    labels = np.zeros(len(students), dtype=int)
    for idx, cluster in enumerate(communities):
        for node in cluster:
            labels[students.index(node)] = idx

    # Ignore trivial cases
    if k < MIN_CLUSTER or k > MAX_CLUSTER:
        continue

    # Compute silhouette score (using distance matrix)
    score = silhouette_score(distance_matrix, labels, metric="precomputed")

    if score > best_score:
        best_score = score
        best_k = k
        best_partition = communities

# === Step 4: Output best result ===
print(f"Best k: {best_k} with silhouette score: {best_score:.4f}")
print("Best partition:", best_partition)
