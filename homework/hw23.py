

# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


data_path ='E:\\python\\ML-day100\\data\\'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()






#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Numeric Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
df.head()






# 對照組 : 標籤編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for i in df.columns:
    df_temp[i] = LabelEncoder().fit_transform(df[i])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')





# 均值編碼 + 邏輯斯迴歸
data = pd.concat([df[:train_num], train_Y], axis=1)
for tt in df.columns:
    mean_df = data.groupby([tt])['Survived'].mean().reset_index()
    mean_df.columns = [tt, f'{tt}_mean']
    data = pd.merge(data, mean_df, on=tt, how='left')
    data = data.drop([tt] , axis=1)
data = data.drop(['Survived'] , axis=1)
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, data, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')

