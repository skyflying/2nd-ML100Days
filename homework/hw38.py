
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')






#讀取 wine 資料集
wine = datasets.load_wine()
wine




test_w=wine.data[:,np.newaxis,2]
print(test_w.shape)




x_train, x_test, y_train, y_test = train_test_split(test_w, wine.target, test_size=0.1, random_state=4)
regress=linear_model.LinearRegression()
regress.fit(x_train,y_train)
y_pred = regress.predict(x_test)



print('Coefficients: ', regress.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))




plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()






#讀取cancer 資料集
breast_cancer = datasets.load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.1, random_state=4)
logreg = linear_model.LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)