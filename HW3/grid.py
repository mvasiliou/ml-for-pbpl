from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

CLASSIFIERS = {}
LARGE_GRID = {}
SMALL_GRID = {}
TEST_GRID = {}

# RANDOM FORESTS
CLASSIFIERS['RF'] = RandomForestClassifier(n_estimators=50, n_jobs=-1)
LARGE_GRID['RF'] = {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]}
SMALL_GRID['RF'] = {'n_estimators':[10, 100], 'max_depth':[5, 50], 'max_features':['sqrt', 'log2'], 'min_samples_split':[2, 10]},
TEST_GRID['RF'] = {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},

# EXTRA TREES
CLASSIFIERS['ET'] = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy')
SMALL_GRID['ET'] = { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,10]}
LARGE_GRID['ET'] = { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]}
TEST_GRID['ET'] = { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]}

# ADA BOOST
CLASSIFIERS['AB'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
SMALL_GRID['AB'] = { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]}
LARGE_GRID['AB'] = { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]}
TEST_GRID['AB'] = { 'algorithm': ['SAMME'], 'n_estimators': [1]}

# LOGIT
CLASSIFIERS['LR'] = LogisticRegression(penalty='l1', C=1e5)
SMALL_GRID['LR'] = { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]}
LARGE_GRID['LR'] = { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]}
TEST_GRID['LR'] = { 'penalty': ['l1'], 'C': [0.01]}

#SVM
CLASSIFIERS['SVM'] = svm.SVC(kernel='linear', probability=True, random_state=0)
SMALL_GRID['SVM'] = {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']}
LARGE_GRID['SVM'] = {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']}
TEST_GRID['SVM'] = {'C' :[0.01],'kernel':['linear']}

# GRADIENT BOOSTING
CLASSIFIERS['GB'] = GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10)
SMALL_GRID['GB'] = {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]}
LARGE_GRID['GB'] = {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]}
TEST_GRID['GB'] = {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]}

# NAIVE BAYES
CLASSIFIERS['NB'] = GaussianNB()
SMALL_GRID['NB'] = {}
LARGE_GRID['NB'] = {}
TEST_GRID['NB'] =  {}

# DECISION TREE
CLASSIFIERS['DT'] = DecisionTreeClassifier()
SMALL_GRID['DT'] = {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]}
LARGE_GRID['DT'] = {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]}
TEST_GRID['DT'] = {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]}

# SGD
CLASSIFIERS['SGD'] = SGDClassifier(loss="hinge", penalty="l2")
SMALL_GRID['SGD'] = { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']}
LARGE_GRID['SGD'] = { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']}
TEST_GRID['SGD'] = { 'loss': ['perceptron'], 'penalty': ['l2']}

# K-NEAREST NEIGHBORS
CLASSIFIERS['KNN'] = KNeighborsClassifier(n_neighbors=3)
SMALL_GRID['KNN'] = {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
LARGE_GRID['KNN'] = {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
TEST_GRID['KNN'] = {'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}

def get_classifiers():
    return CLASSIFIERS

def get_param_grid(grid):
    if grid == 'test':
        return TEST_GRID
    elif grid == 'small':
        return SMALL_GRID
    elif grid == 'large':
        return LARGE_GRID
    else:
        return {}


