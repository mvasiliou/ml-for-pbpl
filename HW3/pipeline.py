import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, ParameterGrid
from grid import *
import datetime



#Plotting the binary variables would probably be more helpful if you grouped your
#data based on another variable. 
#Seeing if the relative frequency of the binary outcomes changes depending on how you filter the data would probably be helpful

##############################################################################
# SETUP FUNCTIONS ############################################################
##############################################################################

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


def setup_results_df():
    '''
    Returns a Dataframe setup to hold prediction results.
    '''
    result_labels = ['classifer', 'parameters', 'auc-roc', 'p_at_5', 'r_at_5',
                     'f1_at_5', 'p_at_10', 'r_at_10', 'f1_at_10,', 'p_at_20', 
                     'r_at_20', 'f1_at_20', 'run_time']
    results_df = pd.DataFrame(columns = result_labels)
    return results_df


def split_testing(X, y, threshold):
    '''
    Splits a given df into a testing and training set

    X: a pandas Dataframe of features
    y: a pandas Dataframe of a target variable
    threshold: a float with the percent of data in the training set

    training_X: a Dataframe of training data for features
    testing_X: a Dataframe of testing data for features
    training_y: a Dataframe of training data for the target
    testing_y: a Dataframe of testing data for the target
    '''
    msk = np.random.rand(len(X)) < threshold
    training_X = X[msk]
    training_y = y[msk]

    testing_X = X[~msk]
    testing_y = y[~msk]

    return training_X, testing_X, training_y, testing_y


def fill_empty(column, fill_method='mean', fill_value=None):
    '''
    Fills missing data with values from the given method.
    '''
    if fill_value:
        column = column.fillna(fill_value)
    elif fill_method == 'mean':
        column = column.fillna(column.mean())
    elif fill_method == 'mode':
        column = column.fillna(column.mode().iloc[0])
    elif fill_method == 'median':
        column = column.fillna(column.median())

    return column

##############################################################################
# FEATURE ENGINEERING FUNCTIONS ##############################################
##############################################################################

def bucket_continuous(df, column, new_column, bins, names=False, left=True, right=True):
    '''
    Splits a continuous column into categorical buckets.
    Modifies the given Dataframe inplace.

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
        df[new_column] = pd.cut(x=df[column],
                                bins=bins,
                                labels=names,
                                include_lowest=left,
                                right=right)


def dummy_categories(df, column):
    '''
    Creates dummy features for a column of categorical data.
    Modifies the given Dataframe inplace.

    df: a pandas Dataframe
    column: the name of the target column 
    '''

    dummies = pd.get_dummies(df[column], prefix='dummy')
    for column in dummies.columns:
        df[column] = dummies[column]

##############################################################################
# CLASSIFIER FUNCTIONS #######################################################
##############################################################################

def classifier_loop(models_to_run, classifiers, grid, X, y):
    '''
    Loops over all of the given classifiers and parameter sets.
    Stores the results of each model in a Dataframe to examine later.

    models_to_run: A list of keys corresponding to the classifiers to use 
    classifiers: A dictionary of classifier methods from sklearn
    grid: A dictionary of parameter sets for each classifier
    X: Dataframe of feature variables
    y: Dataframe of target variable
    '''
    results_df = setup_results_df()
    X_train, X_test, y_train, y_test = split_testing(X, y, threshold = 0.25)
    for label in models_to_run:
        # Set up Classifier
        parameter_values = grid[label]
        classifier = classifiers[label]

        num_params = str(len(ParameterGrid(parameter_values)))
        print('Running ' + label + ' with ' + num_params + ' parameter sets:')

        for p in ParameterGrid(parameter_values):
            try:
                start_time = datetime.datetime.now()

                classifier.set_params(**p)
                fitted_model = classifier.fit(X_train, y_train)
                y_pred_probs = fitted_model.predict_proba(X_test)[:, 1]

                end_time = datetime.datetime.now()
                secs = (end_time - start_time).total_seconds()
                print('\t' + str(secs) + ' seconds with parameters:' + str(p))

                add_results(results_df, label, p, y_test, y_pred_probs, secs)
            except Exception as e:
                print('Error:', e)
    return results_df


##############################################################################
# VALIDATION FUNCTIONS #######################################################
##############################################################################
def add_results(results_df,classifier, p, y_test, y_pred_probs, time):
    '''
    Adds validation values to the results Dataframe, including AUROC,
    precision at k= 5, 10 and 20 and time elapsed.
    '''
    auc = roc_auc_score(y_test, y_pred_probs)

    zipped = zip(y_pred_probs, y_test)
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zipped, reverse = True))

    prec_5, rec_5, f1_5 = prec_recall_k(y_test_sorted, y_pred_probs_sorted, 5)
    prec_10, rec_10, f1_10 = prec_recall_k(y_test_sorted, y_pred_probs_sorted, 10)
    prec_20, rec_20, f1_20 = prec_recall_k(y_test_sorted, y_pred_probs_sorted, 20)

    results = [classifier, p,auc, prec_5, rec_5, f1_5, prec_10, rec_10, f1_10,
               prec_20, rec_20, f1_20, time]

    results_df.loc[len(results_df)] = results
    return results


def generate_binary_at_k(y_scores, k):
    '''
    Generates a list of 1s and 0s based on the length of y_scores and the
    given k.

    y_scores: sorted prediction values
    k: rank of scores to 'retrieve'

    test_predictions_binary: list of 1s and 0s split at cutoff k percent
    '''
    num_scores = len(y_scores)
    cutoff_i = int(num_scores * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_i else 0 for x in range(num_scores)]
    return test_predictions_binary


def prec_recall_k(y_true, y_scores, k):
    '''
    Calculates the precision, recall and F1 scores of predictions 
    at a level k given the true and predicted scores. 

    y_true: actual values of target
    y_scores: predicted values of target
    k: percent of scores to 'retrieve'
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    return precision, recall, f1


def main():
    grid = TEST_GRID
    models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB'] #'SVM','SGD']

    df = read_data("credit-data.csv")
    features  = ['RevolvingUtilizationOfUnsecuredLines',
                  'DebtRatio',
                  'age',
                  'NumberOfTimes90DaysLate',
                  'NumberOfTime30-59DaysPastDueNotWorse',
                  'DebtRatio',
                  'MonthlyIncome',
                  'NumberOfOpenCreditLinesAndLoans',
                  'NumberOfTimes90DaysLate',
                  'NumberRealEstateLoansOrLines',
                  'NumberOfTime60-89DaysPastDueNotWorse',
                  'NumberOfDependents'
                ]

    df['MonthlyIncome'] = fill_empty(df['MonthlyIncome'], 'mean')
    df['NumberOfDependents'] = fill_empty(df['NumberOfDependents'], 'mean')

    X = df[features]
    y = df.SeriousDlqin2yrs

    results_df = classifier_loop(models_to_run, CLASSIFIERS, grid, X, y)
    results_df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()
