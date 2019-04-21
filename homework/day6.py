import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


# �]�w data_path
dir_data = 'E:\\python\\ML-day100\\data\\'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)
app_train.head()


# ���z��ƭȫ������
dtype_select = []
for type in app_train.select_dtypes(exclude=["object"]).dtypes:
    if type not in dtype_select:
        dtype_select.append(type)

numeric_columns = list(app_train.columns[list(app_train.dtypes.isin(dtype_select))])

# �A��u�� 2 �� (�q�`�O 0,1) �����h��
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns : {}".format(len(numeric_columns)))

# �˵��o����쪺�ƭȽd��
for col in numeric_columns:
	print(col," Range �G ",app_train[col].min(),"-",app_train[col].max())
	app_train[col].hist()
	plt.title(col)
	plt.show()
	app_train[[col]].boxplot(vert=False)
	plt.title(col)
	plt.show()
	
	
	

from statsmodels.distributions.empirical_distribution import ECDF
AMT_INCOME_TOTAL=app_train['AMT_INCOME_TOTAL']
print(AMT_INCOME_TOTAL.describe())

# ø�s Empirical Cumulative Density Plot (ECDF)
cdf = ECDF(AMT_INCOME_TOTAL)
plt.plot(cdf.x, cdf.y)
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()

# ���� y �b�� Scale, ���ڭ̥i�H���`�˵� ECDF
plt.plot(np.log(cdf.x[1:]), cdf.y[1:])
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')
plt.show()





# �̤j�ȸ��b�������~
print(app_train['REGION_POPULATION_RELATIVE'].describe())

# ø�s Empirical Cumulative Density Plot (ECDF)
cdf = ECDF(app_train['REGION_POPULATION_RELATIVE'])
plt.plot(cdf.x, cdf.y)
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()

# �N�H�o�����ӻ��A���M����Ʊ��b�����H�~�A�]���ⲧ�`�A�ȥN��o�����q�b�y�L���x���a�Ϧ������I���֡A
# �ɭP region population relative �b�֪����������K���A���b�j�������������|








# �̤j�ȸ��b�������~
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

# ø�s Empirical Cumulative Density Plot (ECDF)
cdf = ECDF(app_train['OBS_60_CNT_SOCIAL_CIRCLE'])
plt.plot(cdf.x, cdf.y)
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()

app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()

#�� histogram �e�X�W���o�ع� (�u�X�{�@���A���O x �b�����ܪ��ɭP�k�䦳�@�j���ťծɡA�N��k�䦳�Ȧ��O�ƶq�}��)
#�o�ɥi�H�Ҽ{�� value_counts �h���o�Ǽƭ�
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False))





# ��@�Ƿ��ݭȼȮɥh���A�bø�s�@�� Histogram
# ��� OBS_60_CNT_SOCIAL_CIRCLE �p�� 20 ������Iø�s
loc_a = app_train['OBS_60_CNT_SOCIAL_CIRCLE'] < 20
loc_b = 'OBS_60_CNT_SOCIAL_CIRCLE'

app_train.loc[loc_a, loc_b].hist()
plt.show()