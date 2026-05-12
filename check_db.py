import sqlite3

conn = sqlite3.connect('instance/users.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Check analysis_history schema
print("\nanalysis_history columns:")
cursor.execute("PRAGMA table_info(analysis_history);")
cols = cursor.fetchall()
for col in cols:
    print(f"  {col[1]} ({col[2]})")

# Check if analysis data exists
print("\nAnalysis records:")
cursor.execute("SELECT id, jd_title, company_name, fit_level, match_score FROM analysis_history;")
records = cursor.fetchall()
for record in records:
    print(f"  ID={record[0]}, Title={record[1]}, Company={record[2]}, Fit={record[3]}, Score={record[4]}")

conn.close()
print("\nDatabase check complete!")
