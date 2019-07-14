import keras
#from keras.datasets import cifar10
from keras.datasets import mnist 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
%matplotlib inline





#�Ĥ@�B�G��ܼҫ�, ���Ǽҫ��O�h�Ӻ����h���u�ʰ��|
 
model = Sequential()

#�ĤG�B�G�c�غ����h
 
model.add(Dense( 500,input_shape=(784,))) # ��J�h�A28*28=784   
model.add(Activation('relu')) # �E����ƬOrelu   

model.add(Dense( 500)) # ���üh�`�I500��   
model.add(Activation('relu'))  

model.add(Dense( 500)) # ���üh�`�I500��   
model.add(Activation('relu'))  

model.add(Dense( 500)) # ���üh�`�I500��   
model.add(Activation('relu'))  

model.add(Dense( 10)) # ��X���G�O10�����O�A�ҥH���׬O10   
model.add(Activation('softmax')) # �̫�@�h��softmax�@���E�����
print("Total Parameters�G%d" % model.count_params())




model.summary()




opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])




(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# �ѩ�mist����J�ƾں��׬O(num, 28 , 28)�A�o�̻ݭn��᭱�����ת������_���ܦ�784��   
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2 ])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2 ])  
Y_train = (np.arange(10) == y_train[:, None]).astype(int)
Y_test = (np.arange(10) == y_test[:, None]).astype(int)




import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
history = model.fit(X_train,Y_train,batch_size = 128, epochs=20, shuffle=True,verbose=2,validation_split=0.3 )





print ( " test set " )
scores = model.evaluate(X_test,Y_test,batch_size=200,verbose= 0)
print ( "" )
#print ( " The test loss is %f " % scores)
print ( " The test loss is %f ", scores)
result = model.predict(X_test,batch_size=200,verbose= 0)

result_max = np.argmax(result, axis = 1 )
test_max = np.argmax(Y_test, axis = 1 )

result_bool = np.equal(result_max, test_max)
true_num = np.sum(result_bool)
print ( "" )
print ( " The accuracy of the model is %f " % (true_num/len(result_bool)))




# history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()