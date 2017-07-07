#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from numpy import mean
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

poi_label = ['poi']

features_list = poi_label + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Number of data points
print("Total number of data points: %i" %len(data_dict))

# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Number of poi: %i" % poi)
print("Number of non-poi: %i" % (len(data_dict) - poi))
       
# Number of features used
all_features = data_dict[data_dict.keys()[0]].keys()
print("There are %i features for each person in the dataset, and %i features \ are used" %(len(all_features), len(features_list)))

# Are there features with many missing values?
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] += 1
print("The number of missing values for each feature: ")
for feature in missing_values:
    print("%s: %i" %(feature, missing_values[feature]))

### Task 2: Remove outliers
def outliersplot(data_set, feature_x, feature_y):
    """
    This function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter( x, y )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()
	
# Visualize data to find outliers
#print(outliersplot(data_dict, 'total_payments', 'total_stock_value'))
#print(outliersplot(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
#print(outliersplot(data_dict, 'salary', 'bonus'))
#print(outliersplot(data_dict, 'total_payments', 'other'))

identity = []
for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        identity.append((person, data_dict[person]['total_payments']))
print("Outlier:")
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

# Find people with financial features "NaN"
fin_nan_dict = {}
for person in data_dict:
    fin_nan_dict[person] = 0
for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            fin_nan_dict[person] += 1
print("Find people with financial features NaN")
#print(sorted(fin_nan_dict.items(), key=lambda x: x[1]))

# Find people with email features "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
print("Find people with email features NaN")
#print(sorted(email_nan_dict.items(), key=lambda x: x[1]))

# Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0
new_features_list = features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Select the best features: Removes all features whose variance is below 80% 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features
from sklearn.feature_selection import f_classif
k = 6
selector = SelectKBest(f_classif, k=6)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print("optimized_features_list")
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Extract from dataset with new features
data1 = featureFormat(my_dataset, optimized_features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio'], sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data1)
new_f_features = scaler.fit_transform(new_f_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def evaluate_clf(clf, features, labels, params, iters=100, test_size=0.3):
   print clf
   acc = []
   pre = []
   recall = []
   grid_search = GridSearchCV(clf, params)
   for i in range(iters):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
   print "accuracy: {}".format(mean(acc))
   print "precision: {}".format(mean(pre))
   print "recall:    {}".format(mean(recall))
   best_params = grid_search.best_estimator_.get_params()
   for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
		
# Provided to give you a starting point. Try a variety of classifiers.


# 1. Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
params = {}

print("=========Evaluating naive bayes=========")
evaluate_clf(nb_clf, features, labels, params)
#print("=========Evaluating naive bayes using dataset with new features=========")
#evaluate_clf(nb_clf, new_f_features, new_f_labels, params)


# 2. KMeans
#from sklearn.cluster import KMeans
#k_clf = KMeans()
#k_params = {'n_clusters': [1, 2, 3, 4, 5], 'tol': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}
#k_params = {'n_clusters': [1], 'tol': [1], 'random_state': [42]}
#print("=========Evaluating k-mean=========")
#evaluate_clf(k_clf, features, labels, k_params)


# 3. Logistic Regression
#from sklearn import linear_model
#from sklearn.pipeline import Pipeline
#lo_clf = Pipeline(steps=[
#        ('scaler', preprocessing.StandardScaler()),
#        ('classifier', linear_model.LogisticRegression())])
#         
#lo_params = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001], \
#            'classifier__C': [0.1, 0.01, 0.001, 0.0001]}

#print("=========Evaluate logistic regression=========")
#evaluate_clf(lo_clf, features, labels, lo_params)


# 4. Support Vector Machine
#from sklearn import svm
#s_clf = svm.SVC()
#s_params = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],\
#           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}    
#print("=========Evaluate svm=========")
#evaluate_clf(s_clf, features, labels, s_params)


# 5. Random Forrest
#from sklearn.ensemble import RandomForestClassifier
#rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)
#print("=========Evaluate Random Forrest=========")
#evaluate_clf(rf_clf, features, labels, s_params)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = nb_clf
#clf = k_clf
#clf = lo_clf
#clf = s_clf
#clf = rf_clf

#final_features_list = optimized_features_list #+ ['msg_to_poi_ratio', 'msg_from_poi_ratio']

PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"

RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#print features
dump_classifier_and_data(clf, my_dataset, optimized_features_list)