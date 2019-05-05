



# 載入套件
import pandas as pd
import numpy as np
import copy
import warnings 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
# 讀取訓練與測試資料
data_path ='E:\\python\\ML-day100\\data\\'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')
df_test = pd.read_csv(data_path + 'house_test.csv.gz')




# 重組資料成為訓練 / 預測用格式
train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()



# 檢查欄位缺值數量 (去掉.head()可以顯示全部)
df.isnull().sum().sort_values(ascending=False).head()





#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')



# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
train_num = train_Y.shape[0]
df.head()




# 空值補 -1, 做線性迴歸
df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()



# 空值補 0
df_0 = df.fillna(0)
train_X = df_0[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()



# 空值補平均值
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()




# 空值補 -1, 搭配最大最小化
df = df.fillna(-1)
df_temp = MinMaxScaler().fit_transform(df)
train_X = df_temp[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()



# 搭配標準化
df_temp = StandardScaler().fit_transform(df)
train_X = df_temp[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()





from sklearn.linear_model import LogisticRegression

ti_train = pd.read_csv(data_path + 'titanic_train.csv')
ti_test = pd.read_csv(data_path + 'titanic_test.csv')

ti_label = ti_train.Survived
ti_ids = ti_test.PassengerId
ti_train = ti_train.drop(['Survived','PassengerId'] , axis = 1)
ti_test = ti_test.drop(['PassengerId'], axis = 1)
ti_df = pd.concat([ti_train,ti_test])
ti_df.head()




ti_numerical_features = np.array( ti_df.columns[ti_df.dtypes != 'object'], dtype = str)
print(f' {len(ti_numerical_features)} Numerical Features : {ti_numerical_features} \n')
ti_df = ti_df[ti_numerical_features]
ti_train_num = len(ti_label)
ti_df.head()




#空白值用-1填，並做logistic regression
ti_df_minus1 = ti_df.fillna(-1)
train_x = ti_df_minus1[:ti_train_num]
LogReg = LogisticRegression()
cross_val_score(LogReg,train_x,ti_label,cv=5).mean()




#空白值用平均值填，並做logistic regression
ti_df_mean = ti_df.fillna(ti_df.mean())
train_x = ti_df_mean[:ti_train_num]
LogReg = LogisticRegression()
cross_val_score(LogReg,train_x,ti_label,cv=5).mean()



#空白值用中位數填，並做logistic regression
ti_df_median = ti_df.fillna(ti_df.median())
train_x = ti_df_median[:ti_train_num]
LogReg = LogisticRegression()
cross_val_score(LogReg,train_x,ti_label,cv=5).mean()



#空白值用-1填，並做logistic regression
ti_df_minus1 = ti_df.fillna(-1)
train_x = ti_df_minus1[:ti_train_num]
LogReg = LogisticRegression()
cross_val_score(LogReg,train_x,ti_label,cv=5).mean()


#空白值用最小最大化填，並做logistic regression
ti_df_MinMax = MinMaxScaler().fit_transform(ti_df.fillna(-1))
train_x = ti_df_MinMax[:ti_train_num]
LogReg = LogisticRegression()
cross_val_score(LogReg,train_x,ti_label,cv=5).mean()


#空白值用標準化填，並做logistic regression
ti_df_Standard = StandardScaler().fit_transform(ti_df.fillna(-1))
train_x = ti_df_Standard[:ti_train_num]
LogReg = LogisticRegression()
cross_val_score(LogReg,train_x,ti_label,cv=5).mean()