import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier

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


def fit_model(df, target, features, method):
    y = df[target]
    X = df[features]
    dt = DecisionTreeClassifier()
    model = dt.fit(X, y)
    return model


def predict_model(df, model, features):
    to_drop = [x for x in df.columns if x not in features]
    df.drop(to_drop, axis=1, inplace=True)
    df['prediction'] = model.predict(df)


def validate_model(df, answers, prediction):
    df['answers'] = answers
    df['correct'] = (df['answers'] == df[prediction])
    return len(df[df['correct']==True]) / len(df)
