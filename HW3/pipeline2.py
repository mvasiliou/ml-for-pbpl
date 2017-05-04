import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, ParameterGrid
from grid import *
import datetime


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in
                               range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision


def add_results(results_df,classifier, p, y_test, y_pred_probs, time):
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse = True))
    auc = roc_auc_score(y_test, y_pred_probs)

    k_prec_5 = precision_at_k(y_test_sorted, y_pred_probs_sorted, 5.0)
    k_prec_10 = precision_at_k(y_test_sorted, y_pred_probs_sorted, 10.0)
    k_prec_20 = precision_at_k(y_test_sorted, y_pred_probs_sorted, 20.0)

    results = [classifier, classifier, p,auc, k_prec_5, k_prec_10, k_prec_20, time]
    results_df.loc[len(results_df)] = results
    return results


def setup_results_df():
    result_labels = ['model_type', 'clf', 'parameters', 'auc-roc', 'p_at_5',
                     'p_at_10', 'p_at_20', 'run_time']
    results_df = pd.DataFrame(columns = result_labels)
    return results_df


def classifier_loop(models_to_run, classifiers, grid, X, y):

    results_df = setup_results_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.25,
                                                        random_state = 0)
    for label in models_to_run:
        # Set up Classifier
        parameter_values = grid[label]
        classifier = classifiers[label]
        print('Running ' + label + ' with ' + str(len(ParameterGrid(parameter_values))) + ' parameter sets:')
        for p in ParameterGrid(parameter_values):
            try:
                start_time = datetime.datetime.now()
                classifier.set_params(**p)
                y_pred_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]
                end_time = datetime.datetime.now()
                secs = (end_time - start_time).total_seconds()
                print('\t' + str(secs) + ' seconds with parameters:' + str(p))
                add_results(results_df, label, p, y_test, y_pred_probs, secs)
            except Exception as e:
                print('Error:', e)
    return results_df


def fill_empty(df, fill_method='mean'):
    '''
    Fills missing data with values from the given method.
    '''

    if fill_method == 'mean':
        df = df.fillna(df.mean())
    return df


def main():
    grid = SMALL_GRID
    models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB', ]#'SVM','SGD']

    df = pd.read_csv("credit-data.csv")
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