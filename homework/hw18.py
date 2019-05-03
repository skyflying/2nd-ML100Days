# ���J�򥻮M��
import pandas as pd
import numpy as np

# Ū���V�m�P���ո��
data_path = 'E:\\python\\ML-day100\\data\\'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape




# ���ո�Ʀ����V�m / �w���ή榡
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()





# �q�X�����쪺�����P�ƶq
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df



#�T�w�u�� int64, float64, object �T��������, ���O�N���W�٦s��T�� list ��
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




# �� : ��� (int) �S�x������ (mean)
df[int_features].mean()





# �Ш̧ǦC�X �T�دS�x���� (int / float / object) x �T�ؤ�k (���� mean / �̤j�� Max / �۲��� nunique) ����l�ާ@



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