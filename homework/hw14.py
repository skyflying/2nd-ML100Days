import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ����ĵ�i�T��
%matplotlib inline
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# �]�w data_path
dir_data = 'E:\\python\\ML-day100\\data\\'


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()



# ��ƾ�z ( 'DAYS_BIRTH'����������� )
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
# �ھڦ~�֤������P�էO (�~�ְ϶� - �ٴڻP�_)
age_data = app_train[['TARGET', 'DAYS_BIRTH']] # subset
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365 # day-age to year-age

# �s����������
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], 
                                  bins = np.linspace(20, 70, num = 11)) #�� 20 �� 70 ���A�� 11 ���I (�o�� 10 ��)
print(age_data['YEARS_BINNED'].value_counts())
age_data.head()







# ��Ƥ��s��Ƨ�
year_group_sorted = np.sort(age_data['YEARS_BINNED'].unique())
age_data.head()

# ø�s���s�᪺ 10 �� KDE ���u
plt.figure(figsize=(8,6))
for i in range(len(year_group_sorted)):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
    
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
plt.title('KDE with Age groups')
plt.show()





# �C�i�Ϥj�p�� 8x8
plt.figure(figsize=(8,8))

# plt.subplot �T�X�p�W�ҭz, ���O��� row�`��, column�`��, ���ϥܲĴX�T(idx)
plt.subplot(321)
plt.plot([0,1],[0,1], label = 'I am subplot1')
plt.legend()

plt.subplot(322)
plt.plot([0,1],[1,0], label = 'I am subplot2')
plt.legend()

plt.subplot(323)
plt.plot([1,0],[0,1], label = 'I am subplot3')
plt.legend()

plt.subplot(324)
plt.plot([1,0],[1,0], label = 'I am subplot4')
plt.legend()

plt.subplot(325)
plt.plot([0,1],[0.5,0.5], label = 'I am subplot5')
plt.legend()

plt.subplot(326)
plt.plot([0.5,0.5],[0,1], label = 'I am subplot6')
plt.legend()

plt.show()



# subplot index �W�L10�H�W��ø�s�覡
nrows = 5
ncols = 2

plt.figure(figsize=(10,30))
for i in range(len(year_group_sorted)):
    plt.subplot(nrows, ncols, i+1)
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 0), 'YEARS_BIRTH'], 
                 label = "TARGET = 0", hist = False)
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 1), 'YEARS_BIRTH'], 
                 label = "TARGET = 1", hist = False)
    plt.title(str(year_group_sorted[i]))
plt.show()  





unique_house_type = app_train.HOUSETYPE_MODE.unique()

nrows = len(unique_house_type)
ncols = nrows // 2

plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):
    plt.subplot(nrows, ncols, i+1)
    app_train.loc[app_train['HOUSETYPE_MODE']==unique_house_type[i], 'AMT_CREDIT'].hist() 
    plt.title(str(unique_house_type[i]))
plt.show()