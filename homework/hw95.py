

import numpy as np
import h5py
import matplotlib.pyplot as plt

# ø�ϵ��G������ܦbJupyter cell ����
%matplotlib inline  
plt.rcParams['figure.figsize'] = (5.0, 4.0) #  �]�wø�ϪO���j�p
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# autoreload�C�i�H���ڭ̤��h�XIPython�N�ʺA�ק�N�X�A�b����N�X�eIPython�|���ڭ̦۰ʭ�����ʪ��Ҷ�
%load_ext autoreload
%autoreload 2
np.random.seed(1)





def zero_pad(X, pad):
    """
    ��image X �� zero-padding. 
    �ѼƩw�q�p�U:
    X -- python numpy array, �e�{���� (m, n_H, n_W, n_C), �N��@�� m �ӹϹ�
         n_H: �ϰ�, n_W: �ϼe, n_C: color channels ��
    pad -- ���, �[�X�骺 zero padding.
    Returns:
    X_pad -- image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C) ����zero-padding �����G
    """
    
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
  
    return X_pad
	
	
	

np.random.seed(1)
x = np.random.randn(4, 5, 5, 2) #����gray image
x_pad = zero_pad(x, 3) # �[��� Pad

print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])
fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])






def pool_forward(A_prev, hparameters, mode = "max"):
    """
    �]�p�@�ӫe����������Ƽh
    �ѼƩw�q�p�U:
    A_prev -- ��J��numpy �}�C, ���� (m, n_H_prev, n_W_prev, n_C_prev)
    hparameter �W�Ѽ� --  "f" and "stride" �ҧΦ���python �r��
    mode -- ���ƪ��Ҧ�: "max" or "average"
    
    ��^:
        A -- ��X�����Ƽh, ���׬� (m, n_H, n_W, n_C) �� numpy �}�C
        cache -- �i�H���Φb backward pass pooling layer ���, �]�t input and hparameter
    """

    # �˯��ؤo from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
 
    # �˯��W�Ѽ� from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # �w�q��X��dimensions
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # ��l�ƿ�X�� matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    ### �{���_�l��m  ###
    for i in range(m): # �V�m�˥���for �j��
        for h in range(n_H): # ��X�˥���for �j��, �w��vertical axis
            for w in range(n_W): #  ��X�˥���for �j��, �w�� horizontal axis
                for c in range (n_C): #  ��X�˥���for �j��, �w��channels

                     # ��X�S�x�Ϫ��e�׸򰪫ץ|���I
                    vert_start = h * stride
                    vert_end = h * stride+ f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # �w�q��i�ӰV�m�ܨ�
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]

                    # �p���Jdata �����Ƶ��G. �ϥ� if statment �h������
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

                        ### �{������  ###
    
    # �x�s��J���S�x�ϸ�ҳ]�w���W�Ѽ�, �i�H�Φb pool_backward()
    cache = (A_prev, hparameters)
    
    # �T�{��X����ƺ���
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache



np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 2}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)