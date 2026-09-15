import pandas as pd
import numpy as np

input_filename = "student_mean_edit_distance_phase3.csv"
attribute_name = "mean_max_edit_distance"
output_filename = "edit_distance_grades_phase3.csv"

# Read the CSV file
df = pd.read_csv(input_filename)

# Ensure edit_distance is numeric
df[attribute_name] = pd.to_numeric(df[attribute_name], errors="coerce")

# Drop NaNs if any edit_distance values were invalid
df = df.dropna(subset=[attribute_name])

# Define reversed score mapping
scores = [10, 8, 6, 4, 2, 0]  # Highest score for lowest distance

# Create bins for equal-width intervals
min_val = df[attribute_name].min()
max_val = df[attribute_name].max()

# 6 equal-width intervals
bins = np.linspace(min_val, max_val, num=7)  # num=7 because bins count = intervals + 1

# Create equal-depth bins (6 bins)
df["score"] = pd.qcut(df[attribute_name], q=6, labels=scores).astype(int)

# Save to a new CSV
df.to_csv(output_filename, index=False)

print(f"Scored CSV saved to {output_filename}")
