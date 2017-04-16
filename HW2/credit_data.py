import pipeline as pl

df = pl.read_data('credit-data.csv')

training, testing, targets = pl.split_testing(df, 0.2, 'SeriousDlqin2yrs')
training = pl.fill_empty(training)
testing = pl.fill_empty(testing)

pl.explore_data(training)

pl.bucket_continuous(training, 'NumberOfDependents', 'dependent_cat', bins=4, names=['Low', 'Okay', 'High', 'Too High'])
pl.dummy_categories(training, 'dependent_cat')

pl.bucket_continuous(testing, 'NumberOfDependents', 'dependent_cat', bins=4, names=['Low', 'Okay', 'High', 'Too High'])
pl.dummy_categories(testing, 'dependent_cat')

features = ['RevolvingUtilizationOfUnsecuredLines',
           'age', 'zipcode', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
           'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
           'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
           'NumberOfTime60-89DaysPastDueNotWorse',
           'dummy_Low', 'dummy_Okay', 'dummy_High',
           'dummy_Too High']

model = pl.fit_model(training, 'SeriousDlqin2yrs', features=features, method='tree')
pl.predict_model(testing, model, features)

accuracy = pl.validate_model(testing, targets, 'prediction')
print(accuracy)