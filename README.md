# Titanic  
---  
![](/fig/Titanic.png)  
- Author: Guo Guanglu  
- E-mail: 2360889142@qq.com  
- QQ: 2360889142    

***
After I have learned the Andrew Ng's coursera about maching learning(ML), I deire to do a small completed ML project. So I begin to look for  projects in the Internet. The Titanic project is an entry-level project and it is a binary classification problem, which is suitable for me to have an understanding of the whole process of machine learning.   
At he begining, I searched the Internet for code written by other people and got a preliminary understanding of the whole process of machine learning. Then according to the workflow, begin to writ  myself code. This is just to implement the entire process of the project. The analysis of specific features, choices, additions or more advanced knowledge still needs to be futher implemented.  

***  
Workflow:  
---  
* Data cleaning    
* Feature engineering   
* Basic model and evalution  
* Hyperparameters tuning   
* Predict results   

***
Data cleaning  
---  
Firstly, we should import dataset and observe first 5 exampeles in dataset.    
```python  
traindata = pd.read_csv(traindata_file)
testdata = pd.read_csv(testdata_file)
print '\n traindata first 5 data: is \n', traindata.head()
```  
![](/fig/fig1.png)  
Next step, we need to know which datas are missing.  
```python
full = pd.concat([traindata, testdata], ignore_index=True)
print '\n Find missing data: \n',full.isnull().sum()
```  
Where we find missing data of trianing dataset and test dataset together.  You will see:  
![](/fig/fig2.png)  
From above results, we can find that 'Embarked' and 'Fare' have little missing data, but 'Cabin' and 'Age' have many missing data.
(not include 'Survived') So we need to fill these missing data.  
**Fill Embarked**:    
According to real meaning, Embarked represents Port of Embarkation, so we use mode of the feature to fill.  
```python
print '\nThe most number of Embarked is \n',full['Embarked'].mode()
full.Embarked.fillna('S', inplace=True)
```  
**Fill fare**:  
Since fare is possiblely related with 'Pclass' according to real conditions, so we can use the fare of the corresponding 'Pclass' of mean value to fill.  
```python  
print '\n Find the data of fare is nan :\n', full[full.Fare.isnull()]
Pclass3_mean = full[full.Pclass==3]['Fare'].mean()
full.Fare.fillna(Pclass3_mean, inplace=True)
```  
**Fill Cabin**:  
Since the value of 'Cabin' itself has no practical significance, and due to missing lots of data, we naturally observe 'Survived' conditions in missing 'Cabin' data and  reserved 'Cabin' data.  
```python
full.loc[full.Cabin.notnull(),'Cabin'] = 1
full.loc[full.Cabin.isnull(), 'Cabin'] = 0
#show() the plot survive - Cabin
pd.pivot_table(full, index=['Cabin'], values=['Survived']).plot.bar(figsize=(8,5))
plt.title('Survival Rate')

#another draw survive-Cabin
cabin = pd.crosstab(full.Cabin, full.Survived)
cabin.rename(index={0:'no cabin', 1:'cabin'}, columns={0.0:'Dead', 1.0:'Survived'}, inplace=True)
print cabin
cabin.plot.bar(figsize=(8,5))
plt.xticks(rotation=0, size='xx-large')
plt.title('Survived Count')
plt.xlabel('')
plt.legend()
```  
we can get following figures:  
![](/fig/fig3.png)  
![](/fig/fig4.png)  
According to the figure, we can find that in no cabin, most peope died, however in cabin, most people survived. So the feature worked and  fill 'Cabin' use 0 or 1 replace nan or not nan.  
**Fill age**:  
Firstly, we use Pearson correlation coefficient，finding that is is not strongly related with other features. So 
```python
print '\nThe correlation of full is\n', full.corr()
```  
Since age may be related to salutation, so we can use the mean value of corresponding salutation to fill.  
We can get the feature of 'Title':  
```python
full['Title'] = full.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
print '\nshow Title information:\n', full.Title.value_counts()
```  
![](/fig/fig5.png)  
Since there are many salutation, so we can observe which salutation represents male,  and which salutation represents female.
```pyhon
print pd.crosstab(full.Title, full.Sex)
```  
![](/fig/fig6.png)  
From the above results, we find that only 'Dr', other salutations all belong to one sex. So we can firstly divide Title into six 
'Mr', 'Miss', 'Mrs', 'Master', 'Rarewoman','Rareman'.  
```python
#find except Dr others belong to one sex,put it to convert
full.loc[(full.Title=='Dr')& (full.Sex=='female'), 'Title'] = 'Dona'
#we divided five types : Mr, Miss, Mrs, Master, Rarewoman, Rareman
convert_name = {'Capt':'Rareman', 'Col':'Rareman','Don':'Rareman','Dona':'Rarewoman',
    'Dr':'Rareman','Jonkheer':'Rareman','Lady':'Rarewoman','Major':'Rareman',
    'Master':'Master','Miss':'Miss','Mlle':'Rarewoman','Mme':'Rarewoman',
    'Mr':'Mr','Mrs':'Mrs','Ms':'Rarewoman','Rev':'Mr','Sir':'Rareman',
    'the Countess':'Rarewoman'}
full.Title = full.Title.map(convert_name)
print "\n**********convert name to title\n"
print full.Title.value_counts()
```  
Then we show the information of these titles:  
```python  
print '\n Information of Master Age\n',full[full.Title=='Master']['Age'].describe()
print '\n Information of Miss Age\n',full[full.Title=='Miss']['Age'].describe()
print '\n Information of Mrs Age\n',full[full.Title=='Mrs']['Age'].describe()
print '\n Information of Mr Age\n',full[full.Title=='Mr']['Age'].describe()
```  
We find the mean value of 'Master' is about 4, little boy, but not girl title. Since girl hadn't married, so 'Miss' may be have many girls, and the mean value of 'Miss' is about 22, the std is about 12, so we can also divide 'Miss' into two parts :one 'Miss',another 'Girl'.  
![](/fig/fig7.png)  
Since the maximum of 'Master' is about 14, so we can divide the age of less than 14 into 'Girl'. For missing age in 'Miss', we can divide 'Parch'==0 into 'Girl'. Because girls commonly need  someone to accompany.  
```python
#function of finding 'Girl'
def girl(aa):
    if (aa.Age!=999)&(aa.Title=='Miss')&(aa.Age<=14):  
        return 'Girl'  
    elif (aa.Age==999)&(aa.Title=='Miss')&(aa.Parch!=0):  
        return 'Girl'  
    else:  
        return aa.Title
```
```python
#convert girl, for nan fill 99, because nan can't compute
full.Age.fillna(999, inplace=True)
full['Title'] = full.apply(girl, axis=1)
print '\nfinish the convertion of title\n',full.Title.value_counts()
```  
After converting, we can get the title as follows:  
![](/fig/fig8.png)  
The we start to fill 'Age'. We using the median values of the corresponding 'Title' to fill.  
```python
#fill Age
Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    full.loc[(full.Age==999)&(full.Title==i),'Age']=full.loc[full.Title==i,'Age'].median()
print full.info()
```
Finally we can get information of the data:  
![](/fig/fig9.png)  
Since 'Name' and 'Ticket' are not meaningful for prediction, the features is removed from the data. Then full is divided into two parts:
train and test data. Finally save cleaning data.  
```python  
full.drop(columns=['Name', 'Ticket'], inplace=True)
train = full[:891]
test = full[891:]
print train.head()
print test.head()
train.to_csv('../data/fill_train.csv', index=False)
test.to_csv('../data/fill_test.csv', index = False)
```

***
Feature engineering  
---  
Since passenerId is not meaningfu, so we firstly remove the feature from data.  
```python
#drop passengerId , the feature no meaning
train = fill_train.drop(columns='PassengerId')
test = fill_test.drop(columns='PassengerId')
```  
Then we need convert categorical variables to numerical variables, because there is no size relationship between different characters, use one-hot to encode.  
```python
#convert categorical variables to numerical variables
train_dummies = pd.get_dummies(train)
test_dummies = pd.get_dummies(test)
test_predict = test_dummies.drop('Survived', axis=1)
print train_dummies.head()
```  
Next step, get train data and label  
```python
#get train data and label
y = train_dummies.Survived
X = train_dummies.drop('Survived', axis=1)
print 'show first 5 data X:\n', X.head()
print 'show first 5 data y:\n', y.head()
```  
![](/fig/fig10.png)  
And we can visulaize these data by PCA  
```python  
#PCA visualization
array_y = np.array(y)
pca = PCA(n_components=2)
scaler_X= StandardScaler().fit(X).transform(X)
newdata = pca.fit_transform(scaler_X)
print "explained_variance_ratio:\n",pca.explained_variance_ratio_
plt.plot(newdata[array_y==1,0], newdata[array_y==1,1],'+g', \
         newdata[array_y == 0, 0], newdata[array_y == 0, 1], 'or')
plt.show()
```
![](/fig/fig11.png)  
Here I don’t preform complex feature engineering, first convertinng the existing features into learnable data, and then implementing the predictive function.  

***  
Basic model and evaluation  
---  
I use SVM model to prediction.  
Firstly we need split train to train and test, because we need test to evaluate our model.  
```python  
#split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,\
                                                    random_state=123,\
                                                    stratify= y)
```  
we need  scale data , later we will compare the scaled and unscaled results, you will find that scaled data improve the accuracy of prediciton.
```python  
#scaler data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Next step, set hyperparameters, since we will use GridSearchCV to find the best parameters.  
```python  
hyperparameters = {'C':[0.001,0.01,0.1,10,12,14],\
                   'gamma':[0.001,0.003,0.005,0.06]}
```
Let's start to learn SVM model  
```python  
clf = GridSearchCV(svm.SVC(), hyperparameters, cv=10)
clf.fit(X_train, y_train)
```
After learning model, we use our divided test data to evaluate our model.  
```python  
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
```
![](/fig/fig12.png)  
From above results, we find scaled resulted is better than unscaled resulted.  
Then we can use learning-curve to observe underfitting or overfitting.  
```python  
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#draw learning_curve
estimator = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
title = "Learning Curves (SVM)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(estimator, title, X_train_scaled, y_train, cv=cv, n_jobs=1)
```
![](/fig/fig13.png)  
From the above figure, we can find litte variance and a larger bias, so our model is underfitting, we need to add some features to improve accuracy.  
Therefore, I use the product of every two items as a new feature.  
```python  
def addProduct_NewFeature(Dataframe):
    temp_Dataframe = Dataframe
    columns_size = Dataframe.iloc[0].size
    temp_Dataframe.columns = range(columns_size)
    product_feature = temp_Dataframe
    for i in range(columns_size):
        temp = temp_Dataframe[i]
        list_temp = list(temp)
        rept_temp = np.array([list_temp]*(columns_size-i)).T
        temp = pd.DataFrame(rept_temp)
        temp2 = temp_Dataframe[range(i,columns_size)]
        temp2.columns = range(columns_size - i)
        temp = temp2 * temp
        product_feature = pd.concat([product_feature, temp], axis=1, ignore_index=True)
    return product_feature
```  
Add new feature  
```python  
X = addProduct_NewFeature(X)
test_predict = addProduct_NewFeature(test_predict)
print X.head()
```  
![](/fig/fig14.png)
The learning steps are the same.  

***  
Predict results  
---  
Finally, we can use our learned model to predict test dataset and save the results according to the requirements.  
```python   
test_predict_scaled = scaler.transform(test_predict)
predict_result = clf.predict(test_predict_scaled)
result = pd.DataFrame({'PassengerId':fill_test.PassengerId, 'Survived': predict_result})
result['Survived'] = result['Survived'].astype('int')
result.to_csv('G.csv', index= False)
```  
Note: the results of prediction  is float-type, we need convert it to int-type  
```python   
result['Survived'] = result['Survived'].astype('int')
result.to_csv('G.csv', index= False)
```   
If you want to save your model, you can:  
```python
joblib.dump(clf, 'svm.pkl')
```
This is my scored in leaderboard of kaggle:  
![](/fig/fig15.png)  

***  
Summary  
---  
A completed ML project includes:  
1.Data cleaing   
2.Visualization (not introducing)  
3.Feature engineering  
4.Basic model and evalution
5.Hyperparameters tuning  
6.Prediction

And we can use learning-curve to observe fitting conditions.  
Overall,through this project, I have a preliminary understanding of the construction of learning projects and I am aware that besides the model, it is also very important for data cleaning and feature engineering.In the future, I will learn about feature selection to improve the accuracy of the model.







***  
Reference  
----
https://blog.csdn.net/wydyttxs/article/details/76695205 about Pearson  
https://blog.csdn.net/lll1528238733/article/details/75114360  about a completed project
https://zhuanlan.zhihu.com/p/27655949 about visualization relationship between features and Survived  
https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic the same function with above link
