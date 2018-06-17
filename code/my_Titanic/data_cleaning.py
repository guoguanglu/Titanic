#************************************import libraries********************************
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tool_function import girl

print '\n**************************IMPORT DATA************************************************\n'
traindata_file = 'D:\\Documents\\machine_learning\\project\\Titanic\\data\\train.csv'
testdata_file  = 'D:\\Documents\\machine_learning\\project\\Titanic\\data\\test.csv'

traindata = pd.read_csv(traindata_file)
testdata = pd.read_csv(testdata_file)
print '\n traindata first 5 data: is \n', traindata.head()
raw_input('Pause...........................\nPlease press enter to continue, start data cleaning')

print '\n***************************START DATA CLEANING******************************************************\n'

#observe data information
#train data 891, test data 418
print traindata.info()
print testdata.info()
#watch missing data
full = pd.concat([traindata, testdata], ignore_index=True)
print '\n Find missing data: \n',full.isnull().sum()
#start to fill nan
#according to the real meaning, we fill Embarked by the most number 'S'
print '\nThe most number of Embarked is \n',full['Embarked'].mode()
full.Embarked.fillna('S', inplace=True)

#fill fare
print '\n Find the data of fare is nan :\n', full[full.Fare.isnull()]
Pclass3_mean = full[full.Pclass==3]['Fare'].mean()
full.Fare.fillna(Pclass3_mean, inplace=True)

#fill Cabin use 0 or 1 replace nan or not nan
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


#fill Age
#age is not strongly related with other feature
print '\nThe correlation of full is\n', full.corr()

full['Title'] = full.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
print '\nshow Title information:\n', full.Title.value_counts()
print pd.crosstab(full.Title, full.Sex)
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

print '\n Information of Master Age\n',full[full.Title=='Master']['Age'].describe()
print '\n Information of Miss Age\n',full[full.Title=='Miss']['Age'].describe()
print '\n Information of Mrs Age\n',full[full.Title=='Mrs']['Age'].describe()
print '\n Information of Mr Age\n',full[full.Title=='Mr']['Age'].describe()

#convert girl, for nan fill 99, because nan can't compute
full.Age.fillna(999, inplace=True)
full['Title'] = full.apply(girl, axis=1)
print '\nfinish the convertion of title\n',full.Title.value_counts()
#fill Age
Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    full.loc[(full.Age==999)&(full.Title==i),'Age']=full.loc[full.Title==i,'Age'].median()
print full.info()

#*********************************************import data cleaning*****************************
full.drop(columns=['Name', 'Ticket'], inplace=True)
train = full[:891]
test = full[891:]
print train.head()
print test.head()
train.to_csv('../data/fill_train.csv', index=False)
test.to_csv('../data/fill_test.csv', index = False)
plt.show()