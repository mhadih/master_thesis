import os
import sys
import psycopg2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.db_config import get_code_recorder_config

assignment_phase = 3

# Connection parameters are read from environment variables / .env
conn = psycopg2.connect(**get_code_recorder_config())

query1 = """
    SELECT user_id, assignment, identifier 
    FROM codetrace
"""
query2 = """
    SELECT id, student_id, first_name, last_name 
    FROM users
"""

# Execute the query and store the result in a pandas DataFrame
traces_df = pd.read_sql_query(query1, conn)
users_df = pd.read_sql_query(query2, conn)

# filter by conditions
assignment_name = ""
if assignment_phase == 1:
    assignment_name = "AP-Spring03-CA6"
elif assignment_phase == 2:
    assignment_name = "AP-Spring03-CA6-phase2"
elif assignment_phase == 3:
    assignment_name = "AP-Spring03-CA6-phase3"
else:
    raise ValueError("Invalid assignment_phase: must be 1, 2, or 3")
traces_df = traces_df[traces_df['assignment'] == assignment_name]
users_df = users_df[users_df['student_id'] != "810000000"]

# Print the DataFrame
print("number of codetrace records:", len(traces_df))

# join two table
df = pd.merge(left=users_df, right=traces_df, how='inner', left_on='id', right_on='user_id')

print(df.head())

df = df.drop(columns=['id', 'assignment'])
df = df.sort_values(by='student_id')
grouped = df.groupby(['student_id', 'first_name', 'last_name', 'identifier'])
aggregated = grouped.agg({ 'user_id': ['count']})

print("number of students: ", len(aggregated))

output_file = ""
if assignment_phase == 1:
    output_file = "output-phase1"
elif assignment_phase == 2:
    output_file = "output-phase2"
elif assignment_phase == 3:
    output_file = "output-phase3"

aggregated.to_csv(output_file + '.csv')

conn.close()