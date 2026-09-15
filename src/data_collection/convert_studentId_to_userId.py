import csv
import os
import sys
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.db_config import get_code_recorder_config

# --- Database Connection Configuration (from environment variables / .env) ---
db_config = get_code_recorder_config()

# --- Read CSV and collect student_ids ---
input_csv_path = 'APS03_CA6_grades.csv'
rows_to_process = []

with open(input_csv_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    header = next(reader)  # skip header if it exists
    for row in reader:
        if len(row) >= 5:
            student_id = row[0]
            grade = row[4]
            rows_to_process.append((student_id, grade))

# --- Connect to PostgreSQL ---
conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# --- Process and write output ---
output_csv_path = 'user_grades.csv'
with open(output_csv_path, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['user_id', 'grades'])  # header

    for student_id, grade in rows_to_process:
        cur.execute("SELECT id FROM users WHERE student_id = %s", (student_id,))
        result = cur.fetchone()
        if result:
            user_id = result[0]
            writer.writerow([user_id, grade])  # write only if user_id found

# --- Cleanup ---
cur.close()
conn.close()