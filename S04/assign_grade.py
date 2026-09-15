import pandas as pd

def calculate_score(count):
    if count < 10:
        return 0
    elif count >= 500:
        return 15
    else:
        return (count - 10) * (15 / (500 - 10))

codetrace_filename = "phase3.csv"
result_filename = "phase3_with_scores.csv"

# Read CSV
df = pd.read_csv(codetrace_filename)  # columns: studentId,count

# Filter rows where identifier == "Text Change"
df = df[df['identifier'] == "Text Change"]

# Calculate score
df['score'] = df['count'].apply(calculate_score)

# Keep only studentId and score
result_df = df[['studentId', 'score']]

# Save result
result_df.to_csv(result_filename, index=False)

print(result_df)
