

import numpy as np

def conv_single_step(a_slice_prev, W, b):
    """
    �w�q�@�h Kernel (����), �ϥΪ��Ѽƻ����p�U
    Arguments:
        a_slice_prev -- ��J��ƪ�����
        W -- �v��, �Q�ϥΦb a_slice_prev
        b -- ���t�Ѽ� 
    Returns:
        Z -- �ưʵ��f�]W�Ab�^���n�b�e�@�� feature map �W�����G
    """

    # �w�q�@�Ӥ������� a_slice and W
    s = a_slice_prev * W
    # �[�`�Ҧ��� "s" �ë��w��Z.
    Z = np.sum(s)
    # Add bias b to Z. �o�O float() ���,
    Z = float(Z + b)

    return Z
	
	

np.random.seed(1)
#�w�q�@�� 4x4x3 �� feature map
a_slice_prev = np.random.randn(5, 5, 3)
W = np.random.randn(5, 5, 3)
b = np.random.randn(1, 1, 1)

#���o�p���,���Z�x�}����
Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)