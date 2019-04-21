import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


# 設定 data_path
dir_data = 'E:\\python\\ML-day100\\data\\'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)
app_train.head()


# 先篩選數值型的欄位
dtype_select = []
for type in app_train.select_dtypes(exclude=["object"]).dtypes:
    if type not in dtype_select:
        dtype_select.append(type)

numeric_columns = list(app_train.columns[list(app_train.dtypes.isin(dtype_select))])

# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns : {}".format(len(numeric_columns)))

# 檢視這些欄位的數值範圍
for col in numeric_columns:
	print(col," Range ： ",app_train[col].min(),"-",app_train[col].max())
	app_train[col].hist()
	plt.title(col)
	plt.show()
	app_train[[col]].boxplot(vert=False)
	plt.title(col)
	plt.show()
	
	
	

from statsmodels.distributions.empirical_distribution import ECDF
AMT_INCOME_TOTAL=app_train['AMT_INCOME_TOTAL']
print(AMT_INCOME_TOTAL.describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
cdf = ECDF(AMT_INCOME_TOTAL)
plt.plot(cdf.x, cdf.y)
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()

# 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
plt.plot(np.log(cdf.x[1:]), cdf.y[1:])
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')
plt.show()





# 最大值落在分布之外
print(app_train['REGION_POPULATION_RELATIVE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
cdf = ECDF(app_train['REGION_POPULATION_RELATIVE'])
plt.plot(cdf.x, cdf.y)
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()

# 就以這個欄位來說，雖然有資料掉在分布以外，也不算異常，僅代表這間公司在稍微熱鬧的地區有的據點較少，
# 導致 region population relative 在少的部分較為密集，但在大的部分較為疏漏








# 最大值落在分布之外
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
cdf = ECDF(app_train['OBS_60_CNT_SOCIAL_CIRCLE'])
plt.plot(cdf.x, cdf.y)
plt.grid(color='g', linestyle='--', linewidth=1, alpha=0.4)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()

app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()

#當 histogram 畫出上面這種圖 (只出現一條，但是 x 軸延伸很長導致右邊有一大片空白時，代表右邊有值但是數量稀少)
#這時可以考慮用 value_counts 去找到這些數值
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False))





# 把一些極端值暫時去掉，在繪製一次 Histogram
# 選擇 OBS_60_CNT_SOCIAL_CIRCLE 小於 20 的資料點繪製
loc_a = app_train['OBS_60_CNT_SOCIAL_CIRCLE'] < 20
loc_b = 'OBS_60_CNT_SOCIAL_CIRCLE'

app_train.loc[loc_a, loc_b].hist()
plt.show()