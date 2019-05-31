import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score







boston = datasets.load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=4)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)



print(regr.coef)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))








# 讀取 Boston 資料做LASOO回歸
boston = datasets.load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=4)
lasso = linear_model.Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
y_pred = lasso.predict(x_test)

print(lasso.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))





# 讀取 Boston 資料做Ridge回歸
boston = datasets.load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=4)
ridge = linear_model.Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
y_pred = regr.predict(x_test)



print(ridge.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))