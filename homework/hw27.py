# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
data_path = '~/Downloads/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()




# 取一個類別型欄位, 與一個數值型欄位, 做群聚編碼

#取cabin欄位，做平均值眾數以及中位數等數值欄位做群聚編碼
df['Cabin'] = df['Cabin'].fillna('None')
df_mean = df.groupby(['Cabin'])['Fare'].mean().reset_index()
df_mode = df.groupby(['Cabin'])['Fare'].apply(lambda x: x.mode()[0]).reset_index()
df_median = df.groupby(['Cabin'])['Fare'].median().reset_index()
df_max= df.groupby(['Cabin'])['Fare'].max().reset_index()
df_min= df.groupby(['Cabin'])['Fare'].min().reset_index()
temp = pd.merge(df_mean, df_mode, how='left', on=['Cabin'])
temp = pd.merge(temp, df_median, how='left', on=['Cabin'])
temp = pd.merge(temp, df_max, how='left', on=['Cabin'])
temp = pd.merge(temp, df_min, how='left', on=['Cabin'])
temp.columns = ['Cabin', 'Cabin_Fare_Mean', 'Cabin_Fare_Mode', 'Cabin_Fare__Median', 'Cabin_Fare_Max', 'Cabin_Fare_Min']
temp




df = pd.merge(df, temp, how='left', on=['Cabin'])
df = df.drop(['Cabin'] , axis=1)
df.head()





#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
df.head()






# 原始特徵 + 邏輯斯迴歸

# 沒有這四個新特徵的 dataframe 稱為 df_temp
df_temp = df.drop(['Cabin_Fare_Mean', 'Cabin_Fare_Mode', 'Cabin_Fare__Median', 'Cabin_Fare_Max', 'Cabin_Fare_Min'] , axis=1)

# 原始特徵 + 線性迴歸
train_X = MMEncoder.fit_transform(df_temp)
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()







# 新特徵 + 邏輯斯迴歸

train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

