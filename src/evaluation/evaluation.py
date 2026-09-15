from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mode
import numpy as np
import pickle
import json
import csv

embeddings_filename = "./student_embeddings/user_jina_embeddings.jsonl"
grades_filename = "./gradesheets/user_grades.csv"

with open("best_clustering.pkl", "rb") as f:
    best_k, best_labels = pickle.load(f)
print("best k:", best_k)

user_ids = []
with open(embeddings_filename, 'r') as f:
    for line in f:
        entry = json.loads(line)
        user_ids.append(int(entry["user_id"]))

# Read the CSV into a dictionary: {user_id: grade}
grades_dict = {}
grades = []
with open(grades_filename, mode='r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        uid = int(row[0])
        grade = float(row[1])
        grades_dict[uid] = grade

for uid in user_ids:
    if uid in grades_dict:
        grades.append(grades_dict[uid])
    else:
        print(f"Warning: user_id {uid} not found in CSV.")
        
grades.sort()

def compute_bin_borders(grades, k):
    """
    Compute borders for k equal-depth bins from a sorted list of grades.
    """
    n = len(grades)
    bin_size = n // k
    borders = []

    for i in range(1, k):
        # Use the last element of the i-th bin as the border
        index = i * bin_size
        if index >= n:
            index = n - 1
        borders.append(grades[index])
    
    # Add the maximum grade as the final upper border
    borders.append(grades[-1])
    return borders

def assign_cluster(grade, borders):
    """
    Assigns a grade to a bin/cluster using the borders.
    """
    for i, border in enumerate(borders):
        if grade <= border:
            return i
    return len(borders) - 1  # In case grade > last border

borders = compute_bin_borders(grades, best_k)
print("Bin borders:", borders)

# --- Assign clusters for user_ids ---
ground_truth_labels = []
for uid in user_ids:
    if uid in grades_dict:
        grade = grades_dict[uid]
        cluster = assign_cluster(grade, borders)
        ground_truth_labels.append(cluster)
    else:
        print(f"Warning: user_id {uid} not found in CSV.")
        
# --- NMI ---
nmi_score = normalized_mutual_info_score(ground_truth_labels, best_labels)
print(f"🧠 Normalized Mutual Information (NMI): {nmi_score:.4f}")

# --- Purity ---
def calculate_purity(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    clusters = np.unique(y_pred)
    total_correct = 0
    for cluster in clusters:
        indices = np.where(y_pred == cluster)
        true_labels_in_cluster = y_true[indices]
        if len(true_labels_in_cluster) == 0:
            continue
        most_common = mode(true_labels_in_cluster).mode
        correct = np.sum(true_labels_in_cluster == most_common)
        total_correct += correct
    return total_correct / len(y_true)

purity_score = calculate_purity(ground_truth_labels, best_labels)
print(f"✅ Purity: {purity_score:.4f}")