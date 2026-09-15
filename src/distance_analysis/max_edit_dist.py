import psycopg2
import pandas as pd
import Levenshtein


conn = psycopg2.connect(
    dbname="code_recorder",
    user="hadi",
    password="***REMOVED***",
    host="localhost",
    port=5432
)

query = """
    SELECT user_id, filename, content, date
    FROM codetrace
    WHERE identifier = 'Text Change'
"""

traces_df = pd.read_sql_query(query, conn)

sorted_df = traces_df.sort_values(by='date')

# Assume df is already in correct chronological order
# Columns: user_id, filename, content

max_distances = []

# Group by user_id and filename, keeping original order
for (user_id, filename), group in sorted_df.groupby(['user_id', 'filename'], sort=False):
    contents = group['content'].tolist()
    max_distance = 0
    for i in range(1, len(contents)):
        dist = Levenshtein.distance(contents[i - 1], contents[i])
        max_distance = max(max_distance, dist)
    max_distances.append({
        'user_id': user_id,
        'filename': filename,
        'max_edit_distance': max_distance
    })

# Result DataFrame
result_df = pd.DataFrame(max_distances)
# 28296
print("maximum edit distance: " + str(result_df['max_edit_distance'].max()))
result_df.to_csv("max_edit_distance.py")
