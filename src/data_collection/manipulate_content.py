import pandas as pd

# Load the CSV file
path = '/home/hadi/code recorder/data/modified_codetrace.csv'
df = pd.read_csv(path)
print("### END LOAD")

# # Add double quotes around each entry in the 'content' column
# df['content'] = df['content'].apply(lambda x: f'"{x}"' if pd.notnull(x) else x)
# print("### END PROCESS")

# # Optional: Save the modified DataFrame back to CSV
# new_path = '/home/hadi/code recorder/data/modified_codetrace.csv'
# df.to_csv(new_path, index=False)

# Print the result to verify
# print(df['content'].head())
print(df.columns.tolist())