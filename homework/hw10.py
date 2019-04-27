import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# �]�w data_path
dir_data = 'E:\\python\\ML-day100\\data\\'



# Ū�������
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)
app_train.shape






# �N�u����حȪ����O�����, �� Label Encoder, �p������Y�Ʈ����o�����i�H�Q�]�t�b��
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# �ˬd�C�@�� column
for col in app_train:
    if app_train[col].dtype == 'object':
        # �p�G�u����حȪ����O�����
        if len(list(app_train[col].unique())) <= 2:
            # �N�� Label Encoder, �H�[�J�����Y���ˬd
            print(col)
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
app_train.head()


# ������Ƭ����`�Ȫ����, �t�~�]�@�����O��, �ñN���`������ন�ŭ� (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# �X�ͤ�� (DAYS_BIRTH) ������� 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])






# �[������Y��
corr = app_train.corr()['TARGET']
print(corr)










corr = corr.drop('TARGET',axis=0).sort_values()
corr_f = pd.concat([corr.head(15),corr.tail(15)])
corr_f




from sklearn.metrics import confusion_matrix

for i in corr_f.index:
    if app_train[i].value_counts().shape[0] == 2:
        display(i)
        display(pd.DataFrame(confusion_matrix(app_train[i],app_train['TARGET']),index=[i+'_0',i+'_1'],columns=['TARGET_0','TARGET_1']))
    else:
        app_train.boxplot(i,'TARGET')
        plt.tight_layout()