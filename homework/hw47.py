from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV



# 使用波士頓房價資料集
digits = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=40)
clf = GradientBoostingRegressor(max_depth=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred))



# 設定要訓練的超參數組合
n_estimators = [100, 200, 300,400,500]
max_depth = [1, 3, 5,7,9]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
grid_search = GridSearchCV(clf, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(grid_result.best_params_)






# 使用最佳參數重新建立模型
clf_best = GradientBoostingRegressor(max_depth=grid_result.best_params_['max_depth'],n_estimators=grid_result.best_params_['n_estimators'])
clf_best.fit(x_train, y_train)
y_pred2 = clf_best.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred2))
