import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# 設定 data_path
dir_data = 'E:\\python\\ML-day100\\data\\'



# 讀取資料檔
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)
app_train.shape






# 將只有兩種值的類別型欄位, 做 Label Encoder, 計算相關係數時讓這些欄位可以被包含在內
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder, 以加入相關係數檢查
            print(col)
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
app_train.head()


# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])






# 觀察相關係數
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