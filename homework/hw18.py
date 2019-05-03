# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = 'E:\\python\\ML-day100\\data\\'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape




# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()





# 秀出資料欄位的類型與數量
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df



#確定只有 int64, float64, object 三種類型後, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
print(f'{len(int_features)} Integer Features : {int_features}\n')
print(f'{len(float_features)} Float Features : {float_features}\n')
print(f'{len(object_features)} Object Features : {object_features}')




# 例 : 整數 (int) 特徵取平均 (mean)
df[int_features].mean()





# 請依序列出 三種特徵類型 (int / float / object) x 三種方法 (平均 mean / 最大值 Max / 相異值 nunique) 的其餘操作



print("Mean values of int type data in titanic df as below")
print(df[int_features].mean())
print("Max values of int type data in titanic df as below")
print(df[int_features].max())
print("number of unique values of int type data in titanic df as below")
print(df[int_features].nunique())
print("Mean values of float type data in titanic df as below")
print(df[int_features].mean())
print("Max values of int type data in titanic df as below")
print(df[float_features].max())
print("number of unique values of float type data in titanic df as below")
print(df[float_features].nunique())
print("Mean values of object type data in titanic df as below")
print(df[object_features].mean())
print("Max values of object type data in titanic df as below")
print(df[object_features].max())
print("number of unique values of objecy type data in titanic df as below")
print(df[object_features].nunique())