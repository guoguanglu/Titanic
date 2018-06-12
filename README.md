# Titanic
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
对于年龄的补充由于它的缺失也很多，
