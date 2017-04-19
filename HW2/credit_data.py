import pipeline as pl

df = pl.read_data('credit-data.csv')

training, testing, targets = pl.split_testing(df, 0.2, 'SeriousDlqin2yrs')
training = pl.fill_empty(training)
testing = pl.fill_empty(testing)

pl.explore_data(training)
pl.scatter_data(training, 'SeriousDlqin2yrs')
pl.scatter_data(training, 'MonthlyIncome')

pl.bucket_continuous(training, 'age', 'age_cat', bins=9, names=['20s', '30s', '40s','50s','60s','70s','80s','90s','100s'])
pl.dummy_categories(training, 'age_cat')

pl.bucket_continuous(testing, 'age', 'age_cat', bins=9, names=['20s', '30s', '40s','50s','60s','70s','80s','90s','100s'])
pl.dummy_categories(testing, 'age_cat')

features = ['RevolvingUtilizationOfUnsecuredLines',
           'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
           'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
           'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
           'NumberOfTime60-89DaysPastDueNotWorse',]

dt_model = pl.fit_model(training, 'SeriousDlqin2yrs', features=features, method='tree')
logit_model = pl.fit_model(training, 'SeriousDlqin2yrs', features=features, method='logit')
neighbor_model = pl.fit_model(training, 'SeriousDlqin2yrs', features=features, method='neighbors')


pl.predict_model(testing, dt_model, features, 'dt_prediction')
pl.predict_model(testing, logit_model, features, 'logit_prediction')
pl.predict_model(testing, neighbor_model, features, 'neighbor_prediction')

dt_accuracy = pl.validate_model(testing, targets, 'dt_prediction')
logit_accuracy = pl.validate_model(testing, targets, 'logit_prediction')
neighbor_accuracy = pl.validate_model(testing, targets, 'neighbor_prediction')

best_prediction, best_k = pl.best_k(training, testing, 'SeriousDlqin2yrs', features, 10, targets)

print("Decision Tree: ", dt_accuracy)
print("Logisitic Regression: ", logit_accuracy) 
print("Nearest Neighbors (k=5): ", neighbor_accuracy)
print("Nearest Neighbors (k="+best_k+'): ', best_prediction)