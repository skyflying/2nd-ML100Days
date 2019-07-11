import numpy as np
 
# Sigmoid ��ƥi�H�N����ȳ��M�g��@�Ӧ�� 0 ��  1 �d�򤺪��ȡC�q�L���A�ڭ̥i�H�N�����Ƭ����v��
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])  
        
# define y for output dataset            
y = np.array([[0,0,1,1]]).T




#seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
#�üƳ]�w���ͺؤl�o�쪺�v����l�ƶ����O�H�����G���A
#���C���}�l�V�m�ɡA�o�쪺�v����l�����G���O�����@�P���C
 
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
# define syn1

iter = 0
#�ӯ��g�����v���x�}����l�ƾާ@�C
#�� ��syn0�� �ӥN�� (�Y����J�h-�Ĥ@�h���h�����v���x�}�^
#�� ��syn1�� �ӥN�� (�Y����J�h-�ĤG�h���h�����v���x�}�^







for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    
    '''
    �s�W
    l2_error �ӭȻ����F���g�����w���ɡ��ᥢ�����ƥءC
    l2_delta �ӭȬ��g�T�H�ץ[�v�᪺���g�������~�t�A���F�T�H�~�t�ܤp�ɡA���������w���~�t�C
    '''
 
    # how much did we miss?
    l1_error = y - l1
 
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    
    # update weights
    syn0 += np.dot(l0.T,l1_delta)
     # syn1 update weights
    
print("Output After Training:")
print(l1)
print("\n\n")
print(l1)

