import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def read_data(source, method='csv'):
    '''
    Converts data in given source into a pandas df.

    source: a string with a filename or url for data retrieval
    method: defines the type of data to access

    df: a pandas Dataframe holding data from the given source
    result: returns API data as a JSON if could not be converted to a Dataframe
    '''
    if method == 'csv':
        df = pd.read_csv(source)
        return df

    elif method == 'json':
        response = requests.get(source)
        result = response.json()
        try:
            df = pd.read_json(result)
            return df
        except:
            return result


def split_testing(df, threshold, target=None):
    '''
    Splits a given df into a testing and training set

    df: a pandas Dataframe
    threshold: a float with the percent of data in the training set
    target: a column name as the target variable

    training: a Dataframe of training data
    testing: a Dataframe of testing data
    targets: a Series of the target column data 
    '''
    msk = np.random.rand(len(df)) < threshold
    training = df[msk]
    testing = df[~msk]
    if target:
        targets = testing[target]
        testing = testing.drop(target, axis=1)
        return training, testing, targets
    return training, testing

def explore_data(df):
    for column in df.columns:
        print(column)
        print(df[column].describe())
        print('########################')


def scatter_data(df, target):
    '''
    Creates a scatter plot comparing each possible feature against the target variable.
    
    df: A pandas Dataframe
    target: a string naming a target column in df
    '''
    for column in df.columns:
        df.plot(x= column, y=target, kind='scatter')
        fig = plt.gcf()
        fig.savefig(column + '_' + target + '_scatter.png')
        plt.close()


def fill_empty(df, fill_method='mean'):
    '''
    Fills missing data with values from the given method.
    '''

    if fill_method == 'mean':
        df = df.fillna(df.mean())
    return df


def bucket_continuous(df, column, new_column, bins, names=False, left=True, right=True):
    '''
    Splits a continuous column into categorical buckets

    df: a pandas Dataframe
    column: the target column
    new_column: the name of the new column
    bins: the number of bins, or, a list of bin names
    names: a list of bin names
    left: boolean, for inclusion on left side
    right: boolean, for inclusion on right side
    '''
    num_bins = bins
    if type(num_bins) != int:
        num_bins = len(bins) - 1

    if names and num_bins != len(names):
        print("Need 1 less name than bins")
    else:
        df[new_column] = pd.cut(x=df[column], bins=bins, labels=names, include_lowest=left, right=right)


def dummy_categories(df, column):
    '''
    Creates dummy features for a column of categorical data

    df: a pandas Dataframe
    column: the name of the target column 
    '''
    dummies = pd.get_dummies(df[column], prefix='dummy')
    for column in dummies.columns:
        df[column] = dummies[column]


def fit_model(df, target, features, method, k=5):
    '''
    Fits a model with a given ML method and Dataframe.

    df: a pandas Dataframe
    target: name of a target column
    features: a list of feature columns
    method: the ML method to utilize
    k: if k is necessary for the ML method, a k parameter

    model: a fitted sklearn model
    '''
    y = df[target]
    X = df[features]
    if method == 'tree':
        classifier = DecisionTreeClassifier()
    elif method == 'logit':
        classifier = LogisticRegression()
    elif method == 'neighbors':
        classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(X, y)
    return model


def predict_model(df, model, features, result_name):
    to_drop = [x for x in df.columns if x not in features]
    df[result_name] = model.predict(df.drop(to_drop, axis=1))


def validate_model(df, answers, prediction):
    df['correct'] = (answers == df[prediction])
    return len(df[df['correct']==True]) / len(df)


def best_k(training, testing, target, features, ceiling, target_answers):
    best_prediction = 0
    best_k = 0
    for i in range(1, ceiling+1):
        neighbor_model = fit_model(training, target, features=features, method='neighbors', k=i)
        column = 'neighbor_prediction_k_'+str(i)
        predict_model(testing, neighbor_model, features, column)
        neighbor_accuracy = validate_model(testing, target_answers, column)
        if neighbor_accuracy > best_prediction:
            best_prediction = neighbor_accuracy
            best_k = i
    return best_prediction, best_k
