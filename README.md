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
* Exploratory Visulization   
* Feature Eenggineering   
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

feature engineering
首先先用直接能用的特征运行整个程序，得出一个初步的结果，然后在进行分析  
这里我先用one-hot编码非数值数据然后进行模型调参

learn model  
我打算使用svm模型去解决这个问题。  






***  
Reference  
----
https://blog.csdn.net/wydyttxs/article/details/76695205 about person  
https://blog.csdn.net/lll1528238733/article/details/75114360
https://zhuanlan.zhihu.com/p/27655949 about visualization relationship between features and Survived  
https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic the same function with above link
