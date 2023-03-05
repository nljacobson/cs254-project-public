import sqlite3
import string
import pandas as pd
import glob
import numpy as np

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


# Coding all the features is going to be a NIGHTMARE....
# all have dif scoring systems, will most likely need to go through individually to re-code/normalize to have comparable variables
#

def insert_data():
    '''Read and insert all data into the db file 
    
    '''
    wave_one_features = ["AID", "S59E", "S45A", "S45C", "S62M", "H1FS11", "H1TS3", "H1PF10", "H1PF14",\
                        "H1PF16", "H1DA2", "H1IR3", "H1SE4", "PC40", "H1EE5", "H1EE7", "PA21", \
                        "PA56", "H1IR12", "H1TO50", "S60K", "S60L", "S60M", "S60N", "S60O", "S62B", \
                         "S62E", "S62O", "S62P", "S62H", "S62Q", "S62R", "H1DA6", "H1GH17", "H1GH18", \
                        "H1GH19", "H1GH20", "H1GH21", "H1GH22", "H1GH28", "H1GH29", "H1GH52", "H1ED2", \
                         "H1ED9", "H1ED7", "H1ED15", "H1ED16", "H1ED17", "H1ED18", "H1ED19", "H1ED20", "H1ED24", \
                        "H1FS1", "H1FS2", "H1FS3", "H1FS4", "H1FS5", "H1FS6", "H1FS7", "H1FS8", "H1FS9", "H1FS10", \
                         "H1FS11", "H1FS12", "H1FS13", "H1FS14", "H1FS15", "H1FS16", "H1FS17", "H1FS18", "H1FS19", \
                        "H1RM14", "H1RF14", "H1WP1", "H1WP2", "H1WP3", "H1WP4", "H1WP5", "H1WP6", "H1WP7", "H1WP10", \
                        "H1WP14", "H1PF1", "H1PF2", "H1PF4", "H1PF5", "H1PF7", "H1PF8", "H1PF10", "H1PF14", "H1PF15", \
                        "H1PF16", "H1PF23", "H1PF24", "H1PF25", "H1PF26", \
                         "H1PF30", "H1PF31", "H1PF32", "H1PF33", "H1PF34", "H1PF35", "H1PF36", "H1TO1", "H1TO3", "H1TO9", \
                        "H1TO12", "H1DS5", "H1DS6", "H1DS7", "H1DS8", "H1DS9", "H1DS10", "H1DS11", "H1DS12", "H1DS13", \
                        "H1DS14", "H1DS15"]

    wave_two_features= ["AID", "H2GH22", "H2GH23", "H2GH24", "H2GH25", "H2GH26", "H2GH27", "H2GH30", "H2GH31", "H2GH45", \
                       "H2ED3", "H2ED5", "H2ED11", "H2ED12", "H2ED13", "H2ED14", "H2ED15", "H2ED16", "H2ED17", "H2SE4", \
                        "H2FS1", "H2FS2", "H2FS3", "H2FS4", "H2FS5", "H2FS6", "H2FS7", "H2FS8", "H2FS9", "H2FS10", \
                       "H2FS11", "H2FS12", "H2FS13", "H2FS14", "H2FS15", "H2FS16", "H2FS17", "H2FS18", "H2FS19", "H2WP1", \
                        "H2WP2", "H2WP3", "H2WP4", "H2WP5", "H2WP6", "H2WP7", "H2WP9", "H2WP10", "H2WP13", "H2WP14", \
                        "H2PF1", "H2PF4", "H2PF5", "H2PF7", "H2PF8", "H2PF9", "H2PF10", "H2PF12", "H2PF13", "H2PF14", \
                        "H2PF15", "H2PF16", "H2PF17", "H2PF21", "H2PF22", "H2PF23", "H2PF24", "H2PF25", "H2PF26", \
                       "H2PF27", "H2PF28", "H2PF29", "H2PF30", "H2PF31", "H2PF32", "H2PF33", "H2PF34", "H2PF35", \
                       "H2TO1", "H2TO11", "H2TO15", "H2TO25", "H2TO26", "H2TO27", "H2TO28", "H2TO29", "H2TO30", "H2DS1", \
                        "H2DS2", "H2DS3", "H2DS4", "H2DS5", "H2DS6", "H2DS7", "H2DS8", "H2DS9", "H2DS10", "H2DS11", "H2DS12", \
                       "H2DS13", "H2DS14", "H2FV1", "H2FV2", "H2FV3", "H2FV4", "H2FV5", "H2FV6", "H2FV7", "H2PR3", "H2PR4", \
                       "H2PR5", "H2EE12", "H2EE14"]
    
    wave_four_outcomes= ["AID", "H4TO5"]
    
    
    #Add the wave one DB 
    df1 = pd.read_csv('data/wave_1/21600-0001-Data.tsv', sep='\t', header=0, low_memory=False, usecols=wave_one_features)
    df2 = pd.read_csv('data/wave_2/21600-0005-Data.tsv', sep='\t', header=0, low_memory=False, usecols=wave_two_features)
    df4 = pd.read_csv('data/wave_4/21600-0022-Data.tsv', sep='\t', header=0, low_memory=False, usecols=wave_four_outcomes)
    
    
    conn = sqlite3.connect(db_file)
    drop_table()
    #db1 = df1.to_sql("ADD_health1.db", con = conn)
    #db2 = df2.to_sql("ADD_health2.db", con = conn)
    #db4 = df4.to_sql("ADD_health4.db", con = conn)
    
    return df1, df2, df4


drop_table()

# create_data_table()
# insert_data()

create_data_table()

data= insert_data()

# +
#need to put all the dataframes together such that they are "grouped" by AID kinda
#each AID is a row and has the feature columns from both feature dfs and the outcomes df
# -

data[0]

data_all_features= pd.merge(data[0], data[1], on="AID", how= "outer")

data_all_features

data_all= pd.merge(data_all_features, data[2], on="AID", how= "right")

data_all1= data_all.replace(r'^\s*$',np.nan, regex=True)
data_all1= data_all1.astype(float)

# +
#H4TO5; 997= legit skip, 998= dont know, 996= refused; these should probs all be changed to NaN
# -

import sklearn as sk
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# +
#to use DecisionTreeClassifier, we have to fill all the NaN values w something, we can use the mean
#we can also just exclude any data w NaN values
#we could also fill w other vals...

for col in data_all1:
    data_all1[col].fillna(data_all1[col].mode()[0], inplace=True)
data_all1
# -

x_train, x_test, y_train, y_test = train_test_split(data_all1.iloc[:, :226], data_all1.iloc[:, 226:], test_size= 0.33, random_state= 0)

decision_tree= DecisionTreeClassifier(random_state= 0)
fitted_data= decision_tree.fit(x_train, y_train)
score= decision_tree.score(x_test, y_test)
score

# +
y_train_f = y_train.to_numpy().reshape(y_train.shape[0])

y_test_f = y_test.to_numpy().reshape(y_test.shape[0])
# -

random_forest=  RandomForestClassifier(n_estimators= 3000, criterion='gini', min_samples_split= 6)
fitted_data_forest= random_forest.fit(x_train, y_train_f)
score_forest= random_forest.score(x_test, y_test_f)
print(score_forest)


























