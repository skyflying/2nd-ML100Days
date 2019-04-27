
import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import warnings 
warnings.filterwarnings('ignore')


# �]�w data_path
dir_data = 'E:\\python\\ML-day100\\data\\'


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s ' %(f_app))
app_train = pd.read_csv(f_app)
app_train.head()



# ��ƾ�z ( 'DAYS_BIRTH'����������� )
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])








# �ھڦ~�֤������P�էO (�~�ְ϶� - �ٴڻP�_)
age_data = app_train[['TARGET', 'DAYS_BIRTH']] # subset
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365 # day-age to year-age

#�� 20 �� 70 ���A�� 11 ���I (�o�� 10 ��)
bin_cut =  [i for i in range(20,70,(70-20)//10)] + [70]
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut) 

# ��ܤ��P�ժ��ƶq
print(age_data['YEARS_BINNED'].value_counts())
age_data.head()








# ø�ϫe���Ƨ� / ����
year_group_sorted = list(age_data['YEARS_BINNED'].value_counts().sort_index().index)

fig = plt.figure(figsize=(10,6))
for i in range(len(year_group_sorted)):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i])+' TARGET 0')
    
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i])+' TARGET 1')
plt.title('KDE with Age groups')
fig.legend(loc='center right')
plt.tight_layout(rect=[0,0,0.75,0.95])
plt.show()








# �p��C�Ӧ~�ְ϶��� Target�BDAYS_BIRTH�P YEARS_BIRTH ��������
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups






plt.figure(figsize = (8, 8))

# �H�~�ְ϶��� x, target �� y ø�s barplot
px = age_data['YEARS_BINNED']
py = age_data['TARGET']
sns.barplot(px, py)

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
