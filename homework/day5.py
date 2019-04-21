import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 砞﹚ data_path
dir_data = 'E:\\python\\ML-day100\\data\\'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)




#戈磞瓃
app_train.describe()


app_train.head()





# **璸衡逆"AMT_CREDIT"キА夹非畉い计程程酶ㄤよ瓜


app_train['AMT_CREDIT'].head()

AMT_CREDIT=app_train['AMT_CREDIT']
print("キА:%.3f" %AMT_CREDIT.mean())
print("夹非畉:%.3f" %AMT_CREDIT.std())
print("い计:%.3f" %AMT_CREDIT.median())
print("程:%.0f" %AMT_CREDIT.max())
print("程:%.0f" %AMT_CREDIT.min())

CREDIT=AMT_CREDIT.hist(grid=False, bins=50, range = (0,1000000),rwidth=1.5)
CREDIT.set_title('Distribution of AMT CREDIT')
CREDIT.set_xlabel('Amount of AMT CREDIT')
CREDIT.set_ylabel('Frequency')





# **璸衡逆"AMT_ANNUITY"キА夹非畉い计程程酶ㄤよ瓜

AMT_ANNUITY = app_train['AMT_ANNUITY']
print("キА:%.3f" %AMT_ANNUITY.mean())
print("夹非畉:%.3f" %AMT_ANNUITY.std())
print("い计:%.3f" %AMT_ANNUITY.median())
print("程:%.0f" %AMT_ANNUITY.max())
print("程:%.0f" %AMT_ANNUITY.min())


plt.figure(figsize = (12,8))
Annuity = AMT_ANNUITY.hist(grid=False,bins = 300, color = 'green')
Annuity.set_title('Distribution of AMT Annuity')
Annuity.set_xlabel('Amount of AMT annuity')
Annuity.set_ylabel('Frequency')



# **璸衡逆"AMT_INCOME"キА夹非畉い计程程酶ㄤよ瓜
AMT_income = app_train['AMT_INCOME_TOTAL']
print("キА:%.3f" %AMT_income.mean())
print("夹非畉:%.3f" %AMT_income.std())
print("い计:%.3f" %AMT_income.median())
print("程:%.0f" %AMT_income.max())
print("程:%.0f" %AMT_income.min())


plt.figure(figsize = (12,8))
INCOME = AMT_income.hist(grid=False, bins = 75,range = (0,1000000),   color = 'blue')
INCOME.set_title('Distribution of AMT INCOME')
INCOME.set_xlabel('Amount of AMT INCOME')
INCOME.set_ylabel('Frequency')
AMT_income = app_train['AMT_INCOME_TOTAL']











# **瓜甶ボ
plt.figure(figsize = (15,10))
Annuity = plt.subplot(3,1,1)
Annuity = AMT_ANNUITY.hist(grid=False, bins = 300, range = (0, 100000), color = 'green')
Annuity.set_title('Distribution of AMT Annuity')
Annuity.set_ylabel('Frequency')


INCOME = plt.subplot(3,1,2)
INCOME = AMT_income.hist(grid=False, bins = 100, range = (0,1000000), color = 'blue')
INCOME.set_title('Distribution of AMT INCOME')
INCOME.set_ylabel('Frequency')


CREDIT = plt.subplot(3,1,3)
CREDIT=AMT_CREDIT.hist(grid=False, bins=50, range = (0,1000000),rwidth=1.5)
CREDIT.set_title('Distribution of AMT CREDIT')
CREDIT.set_xlabel('Amount')
CREDIT.set_ylabel('Frequency')


