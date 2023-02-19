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
    return conn


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
    wave_one_features = ["AID", "S59E", "S45A", "S45C", "S62M", "H1FS11", "H1TS3", "H1PF10", "H1PF14",\
                        "H1PF16", "H1DA2", "H1IR3", "H1SE4", "PC40", "H1EE5", "H1EE7", "PA21", \
                        "PA56", "H1IR12", "H1TO50"]
    #Add the wave one DB 
    df = pd.read_csv('data/wave_1/21600-0001-Data.tsv', sep='\t', header=0, low_memory=False, usecols=wave_one_features)
    
    conn = sqlite3.connect(db_file)
    drop_table()
    db = df.to_sql("ADD_health.db", con = conn)

# create_data_table()
# insert_data()
