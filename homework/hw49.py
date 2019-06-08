import pandas as pd
import numpy as np
import copy, time
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


data_path ='E:\\python\\ML-day100\\data\\'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')


train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()







# 檢查空缺值的狀態
def nan_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))
nan_check(df)




df["Sex"] = df["Sex"].map({"male": 0, "female":1})
df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
df["Age"] = df["Age"].fillna(df['Age'].median())


#Title做分類
df_title = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]
df["Title"] = pd.Series(df_title)
df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df["Title"] = df["Title"].astype(int)
df = pd.get_dummies(df, columns = ["Title"])
# 新建:家庭大小 (Fsize)特徵, 並依照大小分別建獨立欄位
df["Fsize"] = df["SibSp"] + df["Parch"] + 1
df['Single'] = df['Fsize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df['MedF'] = df['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# Ticket : 如果不只是數字-取第一個空白之前的字串(去除'.'與'/'), 如果只是數字-設為'X', 最後再取 One Hot
Ticket = []
for i in list(df.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")        
df["Ticket"] = Ticket
df = pd.get_dummies(df, columns = ["Ticket"], prefix="T")

#Cabin分類
df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in df['Cabin'] ])
df = pd.get_dummies(df, columns = ["Cabin"], prefix="Cabin")

# Embarked, Pclass分類
df = pd.get_dummies(df, columns = ["Embarked"], prefix="Em")
df["Pclass"] = df["Pclass"].astype("category")
df = pd.get_dummies(df, columns = ["Pclass"], prefix="Pc")


# 捨棄 Name
df.drop(labels = ["Name"], axis = 1, inplace = True)


#去除nan值，並用平均取代
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)  
imputer = imputer.fit(df)        
df = imputer.transform(df)


nan_check(df)
df.head()






df = MinMaxScaler().fit_transform(df)
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]






# 線性迴歸預測檔
LR = LogisticRegression()
parameter_LR = dict()
grid_search_LR = GridSearchCV(LR, dict(),scoring='accuracy',cv=10).fit(train_X , train_Y)
LR = grid_search_LR.best_estimator_ 
LR.fit(train_X, train_Y)
LR_pred = LR.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': LR_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('Day_049_submit_titanic_LR.csv', index=False)
print('Best estimator LogisticRegression:',grid_search_LR.best_estimator_  )


# 梯度提升機
gdbt = GradientBoostingClassifier()
parameter_gdbt = dict()
grid_search_gdbt = GridSearchCV(gdbt,  dict(),scoring='accuracy',cv=10).fit(train_X , train_Y)
gdbt = grid_search_gdbt.best_estimator_
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': gdbt_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('Day_049_submit_titanic_gdbt.csv', index=False)
print('Best estimator GradientBoostingClassifier:',grid_search_gdbt.best_estimator_  )


# 隨機森林
RFC = RandomForestClassifier()
parameter_RFC = dict()
grid_search_RFC = GridSearchCV(RFC,  dict(),scoring='accuracy',cv=10).fit(train_X , train_Y)
RFC = grid_search_RFC.best_estimator_
RFC.fit(train_X, train_Y)
RFC_pred = RFC.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': RFC_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('Day_049_submit_titanic_rf.csv', index=False)
print('Best estimator RandomForestClassifier:',grid_search_RFC.best_estimator_  )





# 作業


# LR: 0.66985
# gdbt: 0.60287
# RFC: 0.698561

blending_pred = LR_pred*0.4  + gdbt_pred*0.2 + RFC_pred*0.4
sub = pd.DataFrame({'PassengerId': ids, 'Survived': blending_pred})
sub['Survived'] =  sub['Survived'].map(lambda x:1 if x>0.7 else 0) 
sub.to_csv('Day_049_submit_titanic_blending.csv', index=False)
#Score=0.68899