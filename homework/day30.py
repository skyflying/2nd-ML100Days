# �����S�x�u�{�e���Ҧ��ǳ�
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# �]�����X(fit)�P�s�X(transform)�ݭn���}, �]�����ϥ�.get_dummy, �ӱĥ� sklearn �� OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

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





train_X = df.values
# �]���V�m�޿贵�j�k�ɤ]�n���, �]���N�V�m�Τ����T���� train / val / test, �ĥ� test ���ҦӫD k-fold ��e����
# train �ΨӰV�m��״��ɾ�, val �ΨӰV�m�޿贵�j�k, test ���ҮĪG
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.5)
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.5)





# �H���˪L���X��, �A�N���s�X (*.apply) ���G���W�� / �޿贵�j�k
rf = RandomForestClassifier(n_estimators=20, min_samples_split=10, min_samples_leaf=5, 
                            max_features=4, max_depth=3, bootstrap=True)
onehot = OneHotEncoder()
lr = LogisticRegression(solver='lbfgs', max_iter=1000)

rf.fit(train_X, train_Y)
onehot.fit(rf.apply(train_X))
lr.fit(onehot.transform(rf.apply(val_X)), val_Y)







# �N�H���˪L+���s�X+�޿贵�j�k���G��X

pred_gdbt_lr = lr.predict_proba(onehot.transform(rf.apply(test_X)))[:, 1]
fpr_gdbt_lr, tpr_gdbt_lr, _ = roc_curve(test_Y, pred_gdbt_lr)

# �N�H���˪L���G��X

pred_gdbt = rf.predict_proba(test_X)[:, 1]
fpr_gdbt, tpr_gdbt, _ = roc_curve(test_Y, pred_gdbt)





import matplotlib.pyplot as plt
# �N���Gø��

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_gdbt, tpr_gdbt, label='RF')
plt.plot(fpr_gdbt_lr, tpr_gdbt_lr, label='RF + LR')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()