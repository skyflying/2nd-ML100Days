

from sklearn import datasets, metrics

# 如果是分類問題，請使用 DecisionTreeClassifier，若為回歸問題，請使用 DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split







# 讀取wine資料集
wine = datasets.load_wine()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=4)

# 建立模型
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# DecisionTreeClassifier(criterion=’gini’
# , splitter=’best’
# , max_depth=None
# , min_samples_split=2
# , min_samples_leaf=1
# , min_weight_fraction_leaf=0.0
# , max_features=None
# , random_state=None
# , max_leaf_nodes=None
# , min_impurity_decrease=0.0
# , min_impurity_split=None
# , class_weight=None
# , presort=False)
clf = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier(max_depth = 1)

# 訓練模型
clf.fit(x_train, y_train)
clf2.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)
y_pred2 = clf2.predict(x_test)





acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)
acc2 = metrics.accuracy_score(y_test, y_pred2)
print("Acuuracy: ", acc2)




print(wine.feature_names)






print("Feature importance: ", clf.feature_importances_)
print("Feature importance: ", clf2.feature_importances_)



from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())











wine = datasets.load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=4)
wine_tree = DecisionTreeClassifier()
wine_tree.fit(x_train, y_train)
y_pred = wine_tree.predict(x_test)









acu = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acu)



print(wine.feature_names)
print("Feature importance: ", wine_tree.feature_importances_)





dot_data = StringIO()
export_graphviz(wine_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
