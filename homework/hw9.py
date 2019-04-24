# 載入基礎套件
import numpy as np
np.random.seed(1)

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline







x = np.random.randint(0,50,1000)
y = 50-x+np.random.normal(0,10,1000)
np.corrcoef(x,y) 


plt.scatter(x, y)



#弱相關
x = np.random.randint(0,50,1000)
y = np.random.randint(0,50,1000)
np.corrcoef(x,y) 


plt.scatter(x, y)



＃正相關
x = np.random.randint(0,50,1000)
y = x + np.random.normal(0, 10, 1000)
np.corrcoef(x, y)