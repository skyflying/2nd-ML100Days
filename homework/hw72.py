import numpy as np
from numpy import *
import matplotlib.pylab as plt
%matplotlib inline


def ReLU(x):
    return abs(x)*(x>0)
def dReLU(x):
    return (1*(x>0))
x=plt.linspace(-10,10,100)

plt.plot(x,ReLU(x),'r')
plt.text(6, 8, r'$f(x)=max(0,x)$', fontsize=15)
plt.plot(x,dReLU(x),'b')
plt.text(6, 2, 'x < 0 ,x = 0')
plt.text(6, 1.5, 'x >= 0 ,x = 1')

