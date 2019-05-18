# �����S�x�u�{�e���Ҧ��ǳ� (�P�e�d�ҬۦP)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')


data_path ='E:\\python\\ML-day100\\data\\'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()








# �]���ݭn�����O���P�ƭȫ��S�x���[�J, �G�ϥγ�²�����S�x�u�{
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()







# ��״��ɾ����X��, �N���G�̷ӭ��n�ʥѰ���C�Ƨ� (note : D27�@�~��'Ticket'�O�Ĥ@�W�S�x, 'Age'�O�ƭȯS�x���ƦW�̰���)
estimator = GradientBoostingClassifier()
estimator.fit(df.values, train_Y)
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
feats




# ��l�S�x + ��״��ɾ�
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()



# �����n�ʯS�x + ��״��ɾ� 

high_feature = list(feats[:4].index)

train_X = MMEncoder.fit_transform(df[high_feature])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()




# �[��n�S�x�P�ؼЪ�����
# �Ĥ@�W : Ticket              
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=df['Ticket'], y=train_Y, fit_reg=False)
plt.show()



# �ĤG�W : Name        
sns.regplot(x=df['Name'], y=train_Y, fit_reg=False)
plt.show()




# �s�@�s�S�x�ݮĪG

df['Add_char'] = (df['Ticket'] + df['Name']) / 2
df['Multi_char'] = df['Ticket'] * df['Name']
df['GO_div1p'] = df['Ticket'] / (df['Name']+1) * 2
df['OG_div1p'] = df['Name'] / (df['Ticket']+1) * 2

train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()