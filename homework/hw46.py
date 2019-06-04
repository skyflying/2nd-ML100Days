from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()



# �����V�m��/���ն�
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=4)
# �إ߼ҫ�
clf = GradientBoostingClassifier()
# �V�m�ҫ�
clf.fit(x_train, y_train)
# �w�����ն�
y_pred = clf.predict(x_test)





acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)