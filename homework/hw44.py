from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split






# 讀取鳶尾花資料集
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)
clf = RandomForestClassifier(n_estimators=10, max_depth=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)





acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)



print(iris.feature_names)
print("Feature importance: ", clf.feature_importances_)




# 讀取wine資料集
wine = datasets.load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=4)
clf = RandomForestClassifier(n_estimators=20, max_depth=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)



acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)






print(wine.feature_names)
print("Feature importance: ", clf.feature_importances_)
