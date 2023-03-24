from os.path import isfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from database import create_data_table, insert_data, fetch_features, make_column_boolean
from features import wave_one_features, wave_two_features, wave_four_outcomes


def normalize(df):
    max_value = max(df)
    return 1 / max_value * df


def hypothesis(x, theta):
    h = 1 / (1 + np.exp(-np.matmul(x, theta)))
    # Preventing hypothesis from maxing out
    h[np.where(h == 1)] = .999
    return h


def calcLogRegressionCost(x, y, theta):
    """
    Calculate Logistic Regression Cost

    X: Features matrix
    Y: Output matrix
    theta: matrix of variable weights
    output: return the cost value.
    """
    m = x.shape[0]
    cost = (1 / m) * (- np.matmul(y.transpose(), np.log(hypothesis(x, theta))) - np.matmul((1 - y).transpose(), np.log(1 - hypothesis(x, theta))))
    return cost


def logRegressionGradientDescent(X, Y, theta, eta, iters):
    """
    Performs gradient descent optimization on a set of data

    X: Features matrix
    Y: Output matrix
    theta: matrix of variable weights
    eta: learning rate
    iters: number of times to iterate the algorithm (epochs)
    output: return optimized theta and the cost array for each iteration (epoch).
    """
    m = X.shape[0]
    cost = np.zeros(iters)
    for i in range(iters):
        grad_j = (2 / m) * np.matmul(X.transpose(), (hypothesis(X, theta) - Y))
        theta = theta - eta * grad_j
        cost[i] = calcLogRegressionCost(X, Y, theta)
    return theta, cost


def logistic_regression(features, eta, iters, threshold):
    if 'AID' not in features:
        features.append('AID')
    df1 = pd.read_csv('data/wave_1/21600-0001-Data.tsv', sep='\t', header=0, low_memory=False,
                      usecols=wave_one_features)
    df2 = pd.read_csv('data/wave_2/21600-0005-Data.tsv', sep='\t', header=0, low_memory=False,
                      usecols=wave_two_features)
    df4 = pd.read_csv('data/wave_4/21600-0022-Data.tsv', sep='\t', header=0, low_memory=False,
                      usecols=wave_four_outcomes)
    raw_df = pd.merge(df2, df1, on="AID", how="outer")
    raw_df = pd.merge(raw_df, df4, on="AID", how="outer")  # dataframe containing all data
    target = 'H4TO5'  # Value of smoking question
    df = raw_df.loc[:, features + [target]]
    # for feature in raw_df.columns:
    #     if feature in features:
    #         df.drop(feature, axis=1)
    # for feature in features:
    #     df[feature] = pd.to_numeric(df[feature], errors = 'coerce')
    df.insert(0, 'Ones', 1)  # Add ones column to X
    df.replace(np.NaN, 0, inplace=True)
    df.replace(' ', 0, inplace=True)
    test_balanced_data = False
    if test_balanced_data:
        df_old = df[df[target] < threshold]
        df = df[df[target] > threshold]
        df = df.append(df_old.iloc[0:1000, :])
    y = df[target] > threshold
    df = df.drop(target, axis=1)
    df = df.drop('AID', axis=1)
    x = df.to_numpy().astype(float)
    for i in range(x.shape[1]):
        x[:, i] = normalize(x[:, i])
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33)
    y_train = y_train.to_numpy().reshape(len(y_train), 1).astype(int)
    y_test = y_test.to_numpy().reshape(len(y_test), 1).astype(int)
    # Initialize weights
    theta = np.zeros((x.shape[1], 1)).astype(int)
    theta, cost = logRegressionGradientDescent(x_train, y_train, theta, eta, iters)
    score = 0
    true_pos = 0
    true_neg = 0
    #print(sum(y_test)/len(y_test))
    for i in range(len(y_test)):
        # True positive
        if hypothesis(x_test[i], theta) > .5 and y_test[i]:
            score += 1
            true_pos += 1
        # true negative
        if hypothesis(x_test[i], theta) <= .5 and not y_test[i]:
            score += 1
            true_neg += 1
    clf = LogisticRegression(random_state=0, max_iter = 10000).fit(x_train, y_train.ravel())
    sklearn_guesses = clf.predict_proba(x_test)
    sklearn_score = np.dot(sklearn_guesses[sklearn_guesses > .5], y_test) / len(y_test)
    return score / len(y_test)

def hc_logistic_regression(x_train, y_train, x_test, y_test, eta, iters, threshold):
    x_train = np.nan_to_num(x_train)
    # set with threshold


    # Initialize weights
    theta = np.zeros((x_train.shape[1], 1)).astype(int)
    theta, cost = logRegressionGradientDescent(x_train, y_train, theta, eta, iters)
    score = 0
    true_pos = 0
    true_neg = 0
    for i in range(len(y_test)):
        # True positive
        if hypothesis(x_test[i], theta) > .5 and y_test[i]:
            score += 1
            true_pos += 1
        # true negative
        if hypothesis(x_test[i], theta) <= .5 and not y_test[i]:
            score += 1
            true_neg += 1
    print(score/len(y_test))
    return score / len(y_test)

#score = logistic_regression(wave_one_features, eta=.1, iters=1000, threshold=25)
#print(score)