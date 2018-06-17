#*************************************************import libraries********************************************

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from tool_function import plot_learning_curve
from tool_function import addSquare_NewFeature
from tool_function import addProduct_NewFeature

#*************************************************loading data_cleaning************************************

fill_train = pd.read_csv('../data/fill_train.csv')
fill_test = pd.read_csv('../data/fill_test.csv')
print fill_train.head()
#drop passengerId , the feature no meaning
train = fill_train.drop(columns='PassengerId')
test = fill_test.drop(columns='PassengerId')

#*************************************************Feature engineering******************************************

#convert categorical variables to numerical variables
train_dummies = pd.get_dummies(train)
test_dummies = pd.get_dummies(test)
test_predict = test_dummies.drop('Survived', axis=1)
print train_dummies.head()
#get train data and label
y = train_dummies.Survived
X = train_dummies.drop('Survived', axis=1)
print 'show first 5 data X:\n', X.head()
print 'show first 5 data y:\n', y.head()

#add feature
X = addProduct_NewFeature(X)
test_predict = addProduct_NewFeature(test_predict)
print X.head()

raw_input('addSquareFeature')

#PCA visualization
array_y = np.array(y)
pca = PCA(n_components=2)
scaler_X= StandardScaler().fit(X).transform(X)
newdata = pca.fit_transform(scaler_X)
print "explained_variance_ratio:\n",pca.explained_variance_ratio_
plt.plot(newdata[array_y==1,0], newdata[array_y==1,1],'+g', \
         newdata[array_y == 0, 0], newdata[array_y == 0, 1], 'or')
plt.show()
raw_input('stop')
#**************************************************Model********************************************************

#split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,\
                                                    random_state=123,\
                                                    stratify= y)
#Declare data preprocessing steps

#scaler data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#set hyperparameters

#the first group try to tune hyperparameters
#[10,0.03]
hyperparameters = {'C':[0.001,0.01,0.1,10,12,14],\
                   'gamma':[0.001,0.003,0.005,0.06]}

clf = GridSearchCV(svm.SVC(), hyperparameters, cv=10)
clf.fit(X_train, y_train)

#evaluate

pred = clf.predict(X_train)
print '\n The unscaled train accuracy is :', accuracy_score(y_train, pred)

pred = clf.predict(X_test)
print '\n The unscaled test accuracy is :', accuracy_score(y_test, pred)


clf.fit(X_train_scaled, y_train)

pred = clf.predict(X_train_scaled)
print '\nThe scaled train accuracy is :', accuracy_score(y_train, pred)

pred = clf.predict(X_test_scaled)
print '\nThe scaled test accuracy is :', accuracy_score(y_test, pred)
print '\nThe best first group parameters is\n',clf.best_params_
#draw learning_curve
estimator = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
title = "Learning Curves (SVM)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(estimator, title, X_train_scaled, y_train, cv=cv, n_jobs=1)

#predict
test_predict_scaled = scaler.transform(test_predict)
predict_result = clf.predict(test_predict_scaled)
result = pd.DataFrame({'PassengerId':fill_test.PassengerId, 'Survived': predict_result})
result['Survived'] = result['Survived'].astype('int')
result.to_csv('G.csv', index= False)

#**************************************************RandomForest***********
# hyperparameters2 = {'n_estimators':[50,70,80,90,100,120],'max_depth':[5,8,9,11,12,13], 'max_leaf_nodes':[2,7,11,12]}
# clf2 = GridSearchCV(RandomForestClassifier(),hyperparameters2,cv=10)
# clf2.fit(X_train_scaled, y_train)
# pred2 = clf2.predict(X_train_scaled)
# print '\nThe scaled train accuracy RF is :', accuracy_score(y_train, pred2)
#
# pred2 = clf2.predict(X_test_scaled)
# print '\nThe scaled test accuracy is :', accuracy_score(y_test, pred2)
# print '\nThe best first group parameters RF is\n',clf2.best_params_
#
# predict_result2 = clf2.predict(test_predict_scaled)
# result2 = pd.DataFrame({'PassengerId':fill_test.PassengerId, 'Survived': predict_result2})
# result2['Survived'] = result2['Survived'].astype('int')
# result2.to_csv('G2.csv', index= False)


#****************************************Save Model*****************************************
joblib.dump(clf, 'svm.pkl')
# joblib.dump(clf2, 'RF.pkl')

plt.show()


