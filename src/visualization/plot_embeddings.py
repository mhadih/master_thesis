import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np
import json
import torch

user_embeddings = {}
with open("user_CodeT5_embeddings.jsonl", 'r') as f:
    for line in f:
        entry = json.loads(line)
        user_embeddings[entry["user_id"]] = torch.tensor(entry["embedding"])

def plot_embeddings(embeddings_2d, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig(filename)

all_embeddings = []
for embeddings in user_embeddings.values():
    all_embeddings.append(embeddings)
all_embeddings = np.array(all_embeddings)

# Reduce dimensions with PCA
pca = PCA(n_components=2)
emb_pca = pca.fit_transform(all_embeddings)
plot_embeddings(emb_pca, "PCA Visualization", "pca_plot.png")

# Reduce dimensions with t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_tsne = tsne.fit_transform(all_embeddings)
plot_embeddings(emb_tsne, "t-SNE Visualization", "tsne_plot.png")

# Reduce dimensions with UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
emb_umap = reducer.fit_transform(all_embeddings)
plot_embeddings(emb_umap, "UMAP Visualization", "umap_plot.png")
