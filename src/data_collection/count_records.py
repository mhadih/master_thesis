import psycopg2

# List of top student numbers (student_id from users table)
top_students = ['810102398', '810102408', '810102541', '810102543', '810102427',
                '810102428', '810102442', '810102454', '810102467', '810102469',
                '810102470', '810102588', '810101604', '810199456', '810102480',
                '810102482', '810102483', '810102491', '810102494', '810102495',
                '810102499', '810102550', '810102529']

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="code_recorder",
    user="hadi",
    password="***REMOVED***",
    host="localhost",
    port=5432
)

cur = conn.cursor()

# Query to join codetrace and users table
query = """
    SELECT u.student_id, COUNT(*) 
    FROM codetrace c
    JOIN users u ON c.user_id = u.id
    WHERE u.student_id = ANY(%s)
    GROUP BY u.student_id;
"""

cur.execute(query, (top_students,))
results = cur.fetchall()

# Print result
for student_id, count in results:
    print(f"Student Number {student_id}: {count} records")

# Clean up
cur.close()
conn.close()
