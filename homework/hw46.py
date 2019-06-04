from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()



# ちだ癡絤栋/代刚栋
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=4)
# ミ家
clf = GradientBoostingClassifier()
# 癡絤家
clf.fit(x_train, y_train)
# 箇代代刚栋
y_pred = clf.predict(x_test)





acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)