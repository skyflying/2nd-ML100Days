

import numpy as np
y_pred = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 prediction
y_true = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 ground truth


y_pred




from sklearn import metrics

precision = metrics.precision_score(y_true, y_pred) 
recall  = metrics.recall_score(y_true, y_pred) 
f2=(1+2**2)*((precision*recall)/(2**2*precision + recall))



print("F2: ", f2) 
print("Precision: ", precision)
print("Recall: ", recall)