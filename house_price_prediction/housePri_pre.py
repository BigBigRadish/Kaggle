# -*- coding: utf-8 -*-
'''
@author:Zhukun Luo
Jiangxi university of finance and economics
'''
from tpot import TPOTRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
train_df = pd.read_csv('dataset/train.csv', index_col=0)
test_df = pd.read_csv('dataset/test.csv', index_col=0)
#print(train_df.head())
prices = pd.DataFrame({ "log(price + 1)":np.log1p(train_df["SalePrice"])})#数据平滑，正态化
#prices.hist()
#pl.show()
y_train = np.log1p(train_df.pop('SalePrice'))#result set
#print(train_df.head())
all_df = pd.concat((train_df, test_df), axis=0)#train set，多表连接
#print(all_df.shape)
#print(y_train.head())
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)#将数值类型转换为字符串类型，防止数值干扰
#print(all_df['MSSubClass'].value_counts())
all_dummy_df = pd.get_dummies(all_df)#onehot分类
#print(all_dummy_df.head())
#print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))#查看缺失数据
mean_cols = all_dummy_df.mean()#计算每一列的平均值
all_dummy_df = all_dummy_df.fillna(mean_cols)#填充空值
print(all_dummy_df.isnull().sum().sum())#查看是否还有缺失值
numeric_cols = all_df.columns[all_df.dtypes != 'object']#查看数据类型是numberical的columns
print(numeric_cols)
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std#数据标准化(X-X')/s
print(all_dummy_df.head())
'''
建立model
'''
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
X_train = dummy_train_df.values
X_test = dummy_test_df.values
from sklearn.linear_model import Ridge
ridge = Ridge(15)
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
'''
bagging
params = [1, 10, 15, 20, 25, 30, 40]
test_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
pl.plot(params, test_scores)
pl.title("n_estimator vs CV Error")
pl.show()
'''
#XGboost
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier
params = [1,2,3,4,5,6]
test_scores = []
'''
for param in params:
#xgbclassifier
clf = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.3, # 如同学习率
min_child_weight=1, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=5, # 构建树的深度，越大越容易过拟合
gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=1, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样 
reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#reg_alpha=0, # L1 正则项参数
#scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
#objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
#num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=100, #树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
pl.plot(params, test_scores)
pl.title("max_depth vs CV Error");
pl.show()
'''

#xgbregressor
clf = XGBRegressor( learning_rate=0.1, max_depth=6, min_child_weight=2)  
clf.fit(X_train,y_train,eval_metric='auc')
#设置验证集合 verbose=False不打印过程
y_test= np.expm1(clf.predict(X_test))
print(y_test)
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_test})
submission_df.to_csv('dataset/submission.csv',columns = ['Id','SalePrice'],index = False)


'''
#staking效果一般
gbdt = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                            n_estimators=500,
                                            learning_rate=0.05,
                                            max_depth=8,
                                            subsample=0.8,
                                            min_samples_split=9,
                                            max_leaf_nodes=10)
xgb = XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, silent=False)
lr = LinearRegression()
rfg = RandomForestRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=11, min_samples_split=8,
                                        n_estimators=100)
svr_rbf = SVR(kernel='rbf')
stregr = StackingRegressor(regressors=[gbdt, xgb, lr, rfg], meta_regressor=svr_rbf)
stregr.fit(X_train, y_train)
y_test= np.expm1(stregr.predict(X_test))
print(y_test)
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_test})
submission_df.to_csv('dataset/submission.csv',columns = ['Id','SalePrice'],index = False)
'''
'''
遗传 选最优算法模型，最佳特征
tpot = TPOTRegressor(generations=150, verbosity=2) #迭代150次  
tpot.fit(X_train, y_train)  
print(tpot.score(X_test, y_test))  
tpot.export('pipeline.py')  
'''

