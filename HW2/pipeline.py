import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def read_data(source, method='csv'):
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
    for column in df.columns:
        df.plot(x= column, y=target, kind='scatter')
        fig = plt.gcf()
        fig.savefig(column + '_' + target + '_scatter.png')
        plt.close()


def fill_empty(df, fill_method='mean'):
    if fill_method == 'mean':
        df = df.fillna(df.mean())
    return df


def bucket_continuous(df, column, new_column, bins, names=False, left=True, right=True):
    num_bins = bins
    if type(num_bins) != int:
        num_bins = len(bins) - 1

    if names and num_bins != len(names):
        print("Need 1 less name than bins")
    else:
        df[new_column] = pd.cut(x=df[column], bins=bins, labels=names, include_lowest=left, right=right)


def dummy_categories(df, column):
    dummies = pd.get_dummies(df[column], prefix='dummy')
    for column in dummies.columns:
        df[column] = dummies[column]


def fit_model(df, target, features, method, k=5):
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
