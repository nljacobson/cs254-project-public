import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv 
import features as f 
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer


def plot_all_data_over_time():
    csvs = glob("hc_data/r*")
    NUM_COLORS = len(csvs)
    cm = plt.get_cmap('nipy_spectral')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    csvs.sort()
    for csv in csvs: 
        file = open(csv, "r")
        file.readline()
        fitness = []
        for line in file:
            split_line = line.split(",")
            fitness.append(float(split_line[1]))
        plt.plot(fitness[:134], label = csv[-12:-10])
    plt.legend(title = "N. Features", loc=2, prop={'size': 6}) 
    plt.xlabel("Evolutionary Time")
    plt.ylabel("Fitness (Score)")
    plt.title("Random F. Evolution Over Time")
    plt.savefig("evolution_overtime")

def gen_df():
    '''creates data base'''
    df1 = pd.read_csv('data/wave_1/21600-0001-Data.tsv', sep='\t', header=0, low_memory=False, usecols=f.wave_one_features)
    df2 = pd.read_csv('data/wave_2/21600-0005-Data.tsv', sep='\t', header=0, low_memory=False, usecols=f.wave_two_features)
    df4 = pd.read_csv('data/wave_4/21600-0022-Data.tsv', sep='\t', header=0, low_memory=False, usecols=f.wave_four_outcomes)

    data_all_features= pd.merge(df1, df2, on="AID", how= "outer")
    data_all= pd.merge(data_all_features, df4, on="AID", how= "right")

    data_all1= data_all.replace(r'^\s*$',np.nan, regex=True)
    data_all1= data_all1.astype(float)
    data_all1.loc[data_all1['H4TO5'] >=25, 'H4TO5'] = 30
    data_all1.loc[data_all1['H4TO5'].between(5,25), 'H4TO5'] = 15
    data_all1.loc[data_all1['H4TO5'] <5, 'H4TO5'] = 0

    for col in data_all1:
        data_all1[col].fillna(data_all1[col].mode()[0], inplace=True)
    return data_all1

def regenerate_classifier(genome, seed, hyperparams):
    data = gen_df()
    x_train, x_test, y_train, y_test = train_test_split(data[genome], data.iloc[:, 226:], test_size = 0.33, random_state = seed)                                  
    rf= RandomForestClassifier(random_state= seed,min_samples_leaf=hyperparams[0], min_samples_split=hyperparams[1], max_features= hyperparams[2]) 
    rf.fit(x_train, np.ravel(y_train))
    print(rf.score(x_test, y_test))
    y_pred_test = rf.predict(x_test)
    return y_test, y_pred_test, y_train

def make_conf_matrix(y_test, y_pred_test, size, seed):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    matrix_df= pd.DataFrame(confusion_matrix)
    ax = plt.axes()
    plt.figure()
    plot= sns.heatmap(matrix_df, annot=True, fmt='g', ax=ax, cmap="magma")
    ax.set_title('Confusion Matrix: Decision Tree')
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_xticklabels(['Nonsmoker', 'Casual Smoker', 'Daily Smoker'], rotation=45)
    ax.set_ylabel('True Label', fontsize=15)
    ax.set_yticklabels(['Nonsmoker', 'Casual Smoker', 'Daily Smoker'], rotation=0)
    fig1= plot.get_figure()
    fig1.savefig("hc_plots/dt_seed{}_nfeatures{}_conf_matrix.png".format(seed, size), bbox_inches='tight')

def make_ROC_plot(y_train, y_test):
    pass 

def regenerate_all_classifiers():
    csvs = glob("hc_data/d*")
    seed = int(csvs[0][-5])
    for c in csvs:
        file = open(c, "r")
        line = file.readlines()[-1]
        features = [feature.strip("' ") for feature in line.split("[")[1].split("]")[0].split(",")]
        hyper_params = [int(n) for n in line.split("[")[1].split("]")[1].split(",")[1:]]
        
        y_test, y_pred_test, y_train = regenerate_classifier(features, seed, hyper_params)
        make_conf_matrix(y_test, y_pred_test, c[-12:-10], seed)


regenerate_all_classifiers()