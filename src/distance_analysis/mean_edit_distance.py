import pandas as pd

csv_file = 'max_edit_distance.csv'

df = pd.read_csv(csv_file)

# Ensure the max_edit_distance column is numeric
df['max_edit_distance'] = pd.to_numeric(df['max_edit_distance'], errors='coerce')

df = df.dropna(subset=['max_edit_distance'])

mean_distance = df['max_edit_distance'].mean()

#920.38
print(f"Mean max_edit_distance: {mean_distance}")
