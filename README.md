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
工作流程：  
1. Data cleaning  
2. Exploratory Visulization  
3. Feature Engineering  
4. Basic Model and evalution using by CV  
5. Hyperparameters tuning  
6. Ensemble methods  
首先进行数据的导入  
然后进行数据的观察，尤其要观察缺失数据  
然后对缺失数据根据情况进行适当的处理  
这里Embarked只有2个，并且根据实际意义它跟其他关系不大并且数量少，因此我们简单用众数代替  
对于费用跟船票等级有关，因此我们用相应等级的价格的平均数代替  
由于对与Cabin的补充，由于本身的原因不能用数学统计量去代替，再加上缺失数据过多，因此可以用0 和1 代替缺失和不缺失  
对于年龄的补充由于它的缺失也很多，通过person 想关心那个分析与其他变量相关程度低，因此这个特征有必要对整个数据  
参考它人从姓名的称呼去填充，结合实际很相关  
对于称呼的分类，根据名字进行分类，然后根据得到的称呼与性别比较，可以把前四个比较大的称呼作为value，而其他的值较小所以将同时又发现除dr以外其他的都有属于一类性别，因此将其他类别归为两类Rarewoman ，Rareman  ,又因为master相对与小男孩，因此小女孩。  
先假设little girl都没结婚（一般情况下该假设都成立），所以little girl肯定都包含在Miss里面。little boy（Master）的年龄最大值为14岁，所以相应的可以设定年龄小于等于14岁的Miss为little girl。对于年龄缺失的Miss，可以用(Parch!=0)来判定是否为little girl，因为little girl往往是家长陪同上船，不会一个人去。  使用每title的中位数去表示相应称呼缺失的Age。去掉name和ticket number 得到我们的特征，然后输出数据清洗之后的数据  

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
