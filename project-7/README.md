## Identify Fraud using Enron Emails and Financial Data.

### Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.


__Question 1: *Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? [relevant rubric items: “data exploration”, “outlier investigation”].*__

The goal is to use enron email data to build a prediction model that can effectively classify individuals into POI (person of interest) and non-POI. 

Summary of the data

Total number of data points: 146
Total number of poi: 18
Total number of non-poi: 128
There are 21 features for each person in the dataset, and 20 features are used

The number of missing values for each feature:

| Feature | NaN per feature |
| :--- | :--: |
| Loan advances | 142 |
| Director fees | 129 |
| Restricted stock deferred | 128 |
| Deferred payment | 107 |
| Deferred income | 97 |
| Long term incentive | 80 |
| Bonus | 64 |
| Emails sent also to POI | 60 |
| Emails sent | 60 |
| Emails received | 60 |
| Emails from POI | 60 |
| Emails to POI | 60 |
| Other | 53 |
| Expenses | 51 |
| Salary | 51 |
| Excersised stock option | 44 |
| Restricted stock | 36 |
| Email address | 35 |
| Total payment | 21 |
| Total stock value | 20 |


Scatter plots, show that one outlier is "TOTAL", so it can be removed. "THE TRAVEL AGENCY IN THE PARK" was also removed because it is not a person but an agency. "LOCKHART EUGENE E" record did not have any value other than NaN, thus removed this record too.


__Question 2: *What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]*__

I used VarianceThreshold function to removes all features whose variance is below 80%, then used SelectKBest function to obtain the score of each feature; then sorted those scores and used 6 best features as final features, based on the comparison of validation metrics for different values of k as mentioned below. Then scaled all features using 'MinMaxScaler'. We can see all email and financial data are varied largely. Therefore, it is important that we do feature-scaling for the features to be considered evenly.

| Selected Features       | Score↑ |
| :---------------------- | -----: |
| exercised_stock_options | 24.815 |
| total_stock_value       | 24.182 |
| bonus                   | 20.792 |
| salary                  | 18.289 |
| deferred_income         | 11.458 |
| long_term_incentive     |  9.922 |

Also made two new features named 'msg_from_poi_ratio' and 'msg_to_poi_ratio'; 'msg_from_poi_ratio' shows the ratio a person receives emails from POI, and 'msg_to_poi_ratio' shows the ratio a person sends emails to POI. Maybe POIs are more likely to contact each other than non-POIs; therefore the two new features would be better predictors of POI; however, the scores from SelectKBest function showed opposite. The performance slightly dropped after adding two new features to features list. The following table displays the drop in performance when I used the two engineered features

| Metric		| Without added features		| With added features | 
| :-------------| ------------------------------| -------------------:| 
| Accuracy		| 0.858							| 0.845				  | 
| Precision		| 0.457							| 0.305				  | 
| Recall		| 0.379							| 0.370				  | 


k=10
Evaluating naive bayes model
	accuracy: 0.796976744186
	precision: 0.295933779353
	recall:    0.35299025974
Evaluating naive bayes model using dataset with new features
	accuracy: 0.801162790698
	precision: 0.312472850111
	recall:    0.370062049062

k=9
Evaluating naive bayes model
	accuracy: 0.84488372093
	precision: 0.368396825397
	recall:    0.322076839827
Evaluating naive bayes model using dataset with new features
	accuracy: 0.839069767442
	precision: 0.354546176046
	recall:    0.316009379509

k=8
Evaluating naive bayes model
	accuracy: 0.848372093023
	precision: 0.376361111111
	recall:    0.324100649351
Evaluating naive bayes model using dataset with new features
	accuracy: 0.840465116279
	precision: 0.35037049062
	recall:    0.31216017316

k=7
Evaluating naive bayes model
	accuracy: 0.854761904762
	precision: 0.432977633478
	recall:    0.373191558442
Evaluating naive bayes model using dataset with new features
	accuracy: 0.842619047619
	precision: 0.395617965368
	recall:    0.37384992785

k=6
Evaluating naive bayes model
	accuracy: 0.85880952381
	precision: 0.457334776335
	recall:    0.379957792208
Evaluating naive bayes model using dataset with new features
	accuracy: 0.845238095238
	precision: 0.405531746032
	recall:    0.370238816739

k=5
Evaluating naive bayes model
	accuracy: 0.863571428571
	precision: 0.463333333333
	recall:    0.358541125541
Evaluating naive bayes model using dataset with new features
	accuracy: 0.852619047619
	precision: 0.438571428571
	recall:    0.361572150072

k=4
Evaluating naive bayes model
	accuracy: 0.847692307692
	precision: 0.445119047619
	recall:    0.278857142857
Evaluating naive bayes model using dataset with new features
	accuracy: 0.83925
	precision: 0.413759462759
	recall:    0.305626984127

k=3
Evaluating naive bayes model
	accuracy: 0.839487179487
	precision: 0.458243506494
	recall:    0.297996031746
Evaluating naive bayes model using dataset with new features
	accuracy: 0.841282051282
	precision: 0.442833333333
	recall:    0.321083333333


__Question 3: *What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [relevant rubric item: “pick an algorithm”]*__


I tried 4 algorithms: naive bayes, k-mean, logistic regression and svm. Naive bayes turned out to give best performance; it performed well in both the test set and the final set while the K-means model only performed well in the test set and failed in the final set

| Algorithm				| Accuracy	| Precision	| Recall	| Best Param |
| :---------------------|-----------|-----------|-----------|-----------: |
| Naive bayes			| 0.858		| 0.457		| 0.379		| 			 |
| K-means				| 0.369		| 0.760		| 0.374		| tol=1, n_clusters=5 |
| Logistic regression	| 0.860		| 0.400		| 0.190		| tol=1, C=0.1 |
| SVM					| 0.866		| 0.142		| 0.038		| kernel='linear', C=1, gamma=1 |



__Question 4: *What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). [relevant rubric item: “tune the algorithm”]*__


Parameters tuning refers to the adjustment of the algorithm when training, in order to improve the fit on the test set. Parameter can influence the outcome of the learning process, the more tuned the parameters, the more biased the algorithm will be to the training data. The strategy can be effective but it can also lead to overfitting models.

Learning algorithms (eg. decision trees, random forests, clustering techniques, etc.) require you to set parameters before you use the models. Tuning an algorithm or machine learning technique, can be thought of as process to optimize the parameters that impact the model in order to enable the algorithm to perform the best.

This process can be difficult & hence because of the difficulty of determining what optimal model parameters are, some use complex learning algorithms before experimenting adequately with simpler alternatives with better tuned parameters.

Here first I split the data into train & test sets using train_test_split from sklearn's cross_validation package where the train set is 70% of the data.

I used GridSearchCV function to get the best parameters for the models i.e.:

| naive bayes: 		  | the model is simple and there's no need to specify any parameter|
| kmeans: 			  | n_clusters is set to 5; tol (relative tolerance) is set to 1, and random_state is set to 42|
| logistic regresion: | C (inverse regularization) is set to 0.1; tol (relative tolerance) is set to 1; random_state is set to 42|
| svm: 				  | kernel is 'linear'; C is 1; gamma is 1; random_state is 42|
 

__Question 5:*What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*__

Validation of a model checks how the results of a statistical analysis will generalize to an independent data set. A classic mistake we can easily make is overfitting a model the overfit model performs well on training data but will fail making predictions with unseen data. To avoid this I tuned just a few parameters. 

Function 'evaluate_clf' also does cross validation to split the data into training data and test data 100 times using train_test_split from sklearn's cross_validation package where the train set is 70% of the data, calculate the accuracy, precision, and recall of each iteration and mean of each metric.


__Question 6:*Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]*__

I used accuracy, precision, and recall. The average performance for each is as below. Thus the best algorithm found here is naive bayes.

| Metric	| Performance on test set | Performance on final set|
| :---------| -----------------------|------------------------:| 
| Accuracy	| 0.858	| 0.847|
| Precision	| 0.457	| 0.457|
| Recall	| 0.379	| 0.384|


Accuracy shows how close a predicted value is to the actual value. 
An accuracy of 0.858 means that the proportion of true results (both true positives and true negatives) is 0.858 among the total number of cases.

Precision measures an algorithm's power to classify true positives from all cases that are classified as positives. 
Precision of 0.457 denotes that among the total 100 persons classified as POIs, 45 persons are actually POIs. 

Recall is a metric that measures an algorithm's power to classify true positives over all cases that are actually positives. 
Recall of 0.379 means that among 100 true POIs existing in the dataset, 37 POIs are correctly classified as POIs

### References:
- [Introduction to Machine Learning (Udacity)](https://www.udacity.com/course/viewer#!/c-ud120-nd)
- [MITx Analytics Edge](https://www.edx.org/course/analytics-edge-mitx-15-071x-0)
- [scikit-learn Documentation](http://scikit-learn.org/stable/documentation.html)
