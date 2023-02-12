import sqlite3
import string
import pandas as pd
import glob

db_file = "ADD_health.db"

def drop_table():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS ADD_health")