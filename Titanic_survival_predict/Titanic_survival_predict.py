# -*- coding: utf-8 -*-
'''
@author:Zhukun Luo
Jiangxi university of finance and economics
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv('dataset/train.csv', index_col=0)
test_df = pd.read_csv('dataset/test.csv', index_col=0)
#Survived = pd.DataFrame({"Survived":train_df["Survived"]})
#Survived.hist()
#plt.show()
print(train_df.info())
'''
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
Survived    891 non-null int64
Pclass      891 non-null int64
Name        891 non-null object
Sex         891 non-null object
Age         714 non-null float64
SibSp       891 non-null int64
Parch       891 non-null int64
Ticket      891 non-null object
Fare        891 non-null float64
Cabin       204 non-null object
Embarked    889 non-null object
dtypes: float64(2), int64(4), object(5)
含有缺失字段，Age和Cabin
'''
print(train_df.describe())
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
train_df.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title("获救情况 (1为获救)") # 标题
plt.ylabel("人数")  

plt.subplot2grid((2,3),(0,1))
train_df.Pclass.value_counts().plot(kind="bar")
plt.ylabel("人数")
plt.title("乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(train_df.Survived, train_df.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title("按年龄看获救分布 (1为获救)")

plt.subplot2grid((2,3),(1,0), colspan=2)
train_df.Age[train_df.Pclass == 1].plot(kind='kde')   
train_df.Age[train_df.Pclass == 2].plot(kind='kde')
train_df.Age[train_df.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")# plots an axis lable
plt.ylabel("密度") 
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '2等舱','3等舱'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
train_df.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")  
#plt.show()
y_train = (train_df.pop('Survived'))#train result set
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.head())


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    known_age=pd.DataFrame(known_age)
    known_age=known_age.fillna(0)
    known_age=known_age.as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X =known_age[:, 1:]
    print(X)

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(all_df)
data_train = set_Cabin_type(data_train)
print(data_train.head())
all_dummy_df = pd.get_dummies(data_train)#onehot
print(all_dummy_df.head())
numeric_cols = all_df.columns[all_df.dtypes != 'object']#查看数据类型是numberical的columns
print(numeric_cols)
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std#数据标准化(X-X')/s
print(all_dummy_df.head())

dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
X_train = dummy_train_df.values
#print(X_train)
X_test = dummy_test_df.values
print(X_test)
X_test=pd.DataFrame(X_test)
X_test=X_test.fillna(0)

from sklearn import linear_model



# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf.fit(X_train, y_train)
print(clf)
predictions = clf.predict(X_test)
result = pd.DataFrame({'PassengerId':dummy_test_df.index, 'Survived':predictions.astype(np.int32)})
result.to_csv("dataset/logistic_regression_predictions.csv", index=False)

'''
#效果不是很好
from sklearn.ensemble import BaggingRegressor
if __name__=='__main__':
    clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=True, n_jobs=-1)
    bagging_clf.fit(X_train, y_train)
    predictions = bagging_clf.predict(X_test)
    result = pd.DataFrame({'PassengerId':dummy_test_df.index, 'Survived':predictions.astype(np.int32)})
    result.to_csv("dataset/logistic_regression_predictions.csv", index=False)
'''