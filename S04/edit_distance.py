import os
import sys
import psycopg2
import pandas as pd
import Levenshtein
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.db_config import get_s04_config

result_filename = "student_mean_edit_distance_phase3.csv"

conn = psycopg2.connect(**get_s04_config())

query = """
    SELECT 
        u."studentStaffId",
        c.filename, 
        c.content, 
        c.date
    FROM "CodeTrace" AS c
    JOIN "UserAttributes" AS u 
        ON c."userId" = u."userId"
    WHERE c.identifier = 'Text Change' AND (c.assignment='AP-Spring04-CA6-Phase3' OR c.assignment='AP-Spring04-CA6-phase3')
"""

traces_df = pd.read_sql_query(query, conn)

sorted_df = traces_df.sort_values(by='date')

# Assume df is already in correct chronological order

blocked_student_id = ["810102117", "810003060", "810101389", "810101524", "810103482", "810103525", "810103568", "810103583", "810103602"]
max_distances = []

# Wrap the groupby iterator with tqdm to show progress
for (student_id, filename), group in tqdm(
    sorted_df.groupby(['studentStaffId', 'filename'], sort=False),
    total=sorted_df.groupby(['studentStaffId', 'filename']).ngroups,
    desc="Processing files"
):
    if student_id in blocked_student_id:
        continue
    contents = group['content'].tolist()
    max_distance = 0
    for i in range(1, len(contents)):
        dist = Levenshtein.distance(contents[i - 1], contents[i])
        max_distance = max(max_distance, dist)
    max_distances.append({
        'student_id': student_id,
        'max_edit_distance': max_distance
    })

# Convert to DataFrame and compute mean per user
df_max = pd.DataFrame(max_distances)
mean_max_per_student = df_max.groupby('student_id')['max_edit_distance'].mean().reset_index()
mean_max_per_student.rename(columns={'max_edit_distance': 'mean_max_edit_distance'}, inplace=True)

# Result DataFrame
result_df = pd.DataFrame(mean_max_per_student)
# 
print("maximum edit distance: " + str(df_max['max_edit_distance'].max()))
mean_max_per_student.to_csv(result_filename, index=False)
