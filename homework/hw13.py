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
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
app_train.head()





# ������Ƭ����`�Ȫ����, �t�~�]�@�����O��, �ñN���`������ন�ŭ� (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# �X�ͤ�� (DAYS_BIRTH) ������� 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])






app_train['DAYS_BIRTH_YEAR_DISCRETE']= pd.cut(app_train['DAYS_BIRTH']/365,12)
app_train['DAYS_BIRTH_YEAR_DISCRETE'].value_counts()


import seaborn as sns
sns.barplot(app_train['DAYS_BIRTH_YEAR_DISCRETE'],app_train['TARGET'])
plt.xticks(rotation=45)




app_train['DAYS_EMPLOYED']= abs(app_train['DAYS_EMPLOYED'])
app_train['DAYS_EMPLOYED_Years_DISCRETE']= pd.cut(app_train['DAYS_EMPLOYED']/365,6)
app_train['DAYS_EMPLOYED_Years_DISCRETE'].value_counts()


sns.barplot(app_train['DAYS_EMPLOYED_Years_DISCRETE'],app_train['TARGET'])
plt.xticks(rotation=45)


