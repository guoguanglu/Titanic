import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve


def girl(aa):
    if (aa.Age!=999)&(aa.Title=='Miss')&(aa.Age<=14):  
        return 'Girl'  
    elif (aa.Age==999)&(aa.Title=='Miss')&(aa.Parch!=0):  
        return 'Girl'  
    else:  
        return aa.Title
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

def addSquare_NewFeature(Dataframe):
    square_Feature = Dataframe**2
    data_concat = pd.concat([Dataframe, square_Feature], axis=1, ignore_index=True)
    return data_concat
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




# a = pd.DataFrame({0:[1,2,3],1:[4,5,6]})
# list_a = list(a[0])
# n = a.columns.size
# rept_a = np.array([list_a]*n).T
# b = pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
# b.columns=range(n)
# print addProduct_NewFeature(b)

