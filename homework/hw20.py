# �����S�x�u�{�e���Ҧ��ǳ� (�P�e�d�ҬۦP)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# ����ĵ�i�T��
import warnings
warnings.filterwarnings('ignore')

data_path ='E:\\python\\ML-day100\\data\\'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id', 'SalePrice'] , axis=1)
df.head()



#�u�� int64, float64 ��ؼƭȫ����, �s�� num_features ��
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')





# �d���r�����, �u�Ѽƭȫ����
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]
df.head()







# ��� 1stFlrSF �P�ؼЭȪ����G��
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
plt.show()

# ���u�ʰj�k, �[�����
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()






# �N 1stFlrSF ����b�Aı�o�A�X���d��, �վ����s��
df['1stFlrSF'] = df['1stFlrSF'].clip(500, 2200)
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

# ���u�ʰj�k, �[�����
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()







# �N 1stFlrSF ����b�Aı�o�A�X���d��, �˱����s��
keep_indexs = (df['1stFlrSF']> 500) & (df['1stFlrSF']< 2200)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

# ���u�ʰj�k, �[�����
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

