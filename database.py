import sqlite3
import string
import pandas as pd
import glob

db_file = "ADD_health.db"

def drop_table():
    ''' Drops existing db file
    '''
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS ADD_health")


def create_data_table():
    ''' Creates empty db
    '''
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE ADD_health (AID NUMERIC)")


def fetch_data():
    ''' returns all rows in db file
    '''
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM ADD_health").fetchall()
    df = pd.DataFrame(rows)
    return df


def insert_column(col_name):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE ADD_health ADD {} NUMERIC".format(col_name))


def insert_data():
    '''Read and insert all data into the db file 
    '''
    pass 
