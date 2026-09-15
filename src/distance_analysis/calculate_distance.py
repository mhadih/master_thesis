import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pickle

embeddings_filename = "user_CodeT5_file_embeddings.jsonl"
grades_filename = "user_grades.csv"
output_csv_filename = "student_distance_matrix.csv"
output_figure_filename = "student_distance_matrix.png"

user_embeddings = {}
with open(embeddings_filename, 'r') as f:
    for line in f:
        entry = json.loads(line)
        user_embeddings[entry["user_id"]] = np.array(entry["embedding"], dtype=np.float32)

# Collect all embeddings into one NumPy array
student_ids = list(user_embeddings.keys())

# Read the CSV into a dictionary: {user_id: grade}
grades_dict = {}
with open(grades_filename, mode='r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        uid = int(row[0])
        grade = float(row[1])
        grades_dict[uid] = grade

# --- Assign grades for student_ids ---
ground_truth_labels = []
embeddings = []
for sid in student_ids[:]:
    sid = int(sid)
    if sid in grades_dict:
        grade = grades_dict[sid]
        ground_truth_labels.append(grade)
        embeddings.append(user_embeddings[str(sid)])
    else:
        print(f"Warning: user_id {sid} not found in grade CSV.")
        student_ids.remove(str(sid))

all_embeddings = np.array(embeddings)

# === Compute cosine similarity matrix ===
similarity_matrix = cosine_similarity(all_embeddings)

# === Convert to [0,1] distance matrix ===
distance_matrix = (1 - similarity_matrix) / 2

# === Normalize distances to [0,1] range based on dataset ===
min_val = distance_matrix.min()
max_val = distance_matrix.max()
normalized_distance = (distance_matrix - min_val) / (max_val - min_val)

# === Store in DataFrame ===
df_dist = pd.DataFrame(normalized_distance, index=student_ids, columns=student_ids)

# Save to CSV
df_dist.to_csv(output_csv_filename)

# === Plot heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(df_dist, annot=False, cmap="viridis", cbar_kws={'label': 'Distance [0-1]'})
plt.title("Student Distance Matrix (Cosine-based)", fontsize=14)
plt.xlabel("Student")
plt.ylabel("Student")
plt.tight_layout()
plt.show()

# Save as PNG
plt.savefig(output_figure_filename, dpi=300)
plt.close()  # close the figure to free memory