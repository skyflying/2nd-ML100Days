import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# �ؼШ��:y=(x+3)^2
def func(x): 
    return np.square(x+3)
    
# �ؼШ�Ƥ@���ɼ�:dy/dx=2*(x+3)
def dfunc(x): 
    return 2 * (x+3)

def GD(w_init, df, epochs, lr):    
    """  ��פU���k�C���w�_�l�I�P�ؼШ�ƪ��@���ɨ�ơA�D�bepochs�����йB�⤤x����s��
        :param w_init: w��init value    
        :param df: �ؼШ�ƪ��@���ɨ��    
        :param epochs: ���йB��g��    
        :param lr: �ǲ߲v    
        :return: x�b�C�����йB��᪺��m   
     """    
    xs = np.zeros(epochs+1) # �� "epochs+1" �নdtype=np.float32    
    x = w_init    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v���x�n��X���T��        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

# �_�l�v��
w_init = 3    
# ����g����
epochs = 20 
# �ǲ߲v   
lr = 0.01   
# ��פU���k 
x = GD(w_init, dfunc, epochs, lr=lr) 
print (x)

#���X���u��
color = 'r'    
 
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))    
plt.scatter(x, func(x), c=color, )    
plt.legend()

plt.show()









line_x = np.linspace(-5, 5, 100)
line_y = func(line_x)
plt.figure('Gradient Desent: Learning Rate')

w_init = 3
epochs = 5
x = w_init
lr = [0.001, 0.01, 0.1]

color = ['r', 'g', 'y']
size = np.ones(epochs+1) * 10
size[-1] = 70
for i in range(len(lr)):
    x = GD(w_init, dfunc, epochs, lr=lr[i])
    plt.subplot(1, 3, i+1)
    plt.plot(line_x, line_y, c='b')
    plt.plot(x, func(x), c=color[i], label='lr={}'.format(lr[i]))
    plt.scatter(x, func(x), c=color[i])
    plt.legend()
plt.show()







def GD_decay(w_init, df, epochs, lr, decay):
    xs = np.zeros(epochs+1)
    x = w_init
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # �ǲ߲v�I�� 
        lr_i = lr * 1.0 / (1.0 + decay * i)
        # v���x�n��?���T��
        v = - dx * lr_i
        x += v
        xs[i+1] = x
    return xs



line_x = np.linspace(-5, 5, 100)
line_y = func(line_x)
plt.figure('Gradient Desent: Decay')

lr = 1.4
iterations = np.arange(300)
decay = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.9]
for i in range(len(decay)):
    decay_lr = lr * (1.0 / (1.0 + decay[i] * iterations))
    plt.plot(iterations, decay_lr, label='decay={}'.format(decay[i]))

plt.ylim([0, 1.1])
plt.legend(loc='best')
plt.show()







line_x = np.linspace(-5, 5, 100)
line_y = func(line_x)
plt.figure('Gradient Desent: Decay')

x_start = -1
epochs = 10

lr = [0.01, 0.3, 0.7, 0.99]
decay = [0.0, 0.01, 0.5, 0.9]

color = ['k', 'r', 'g', 'y']

row = len(lr)
col = len(decay)
size = np.ones(epochs + 1) * 10
size[-1] = 70
for i in range(row):
     for j in range(col):
        x = GD_decay(x_start, dfunc, epochs, lr=lr[i], decay=decay[j])
        plt.subplot(row, col, i * col + j + 1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x, func(x), c=color[i], label='lr={}, de={}'.format(lr[i], decay[j]))
        plt.scatter(x, func(x), c=color[i], s=size)
        plt.legend(loc=0)
plt.show()

