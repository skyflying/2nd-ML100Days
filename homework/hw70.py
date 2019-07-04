from keras.utils import np_utils
import numpy as np
import warnings
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
warnings.filterwarnings("ignore")
np.random.seed(10)


(x_train_image,y_train_label),\
(x_test_image,y_test_label)= mnist.load_data()



x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

#§âLABELÂà¦¨NUMERICAL Categorical 
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)



model = Sequential()



model.add(Dense(units=256, 
                input_dim=784, 
                kernel_initializer='normal', 
                activation='relu'))
				
				
model.add(Dense(units=128, 
                kernel_initializer='normal', 
                activation='relu'))
				
				

model.add(Dense(units=10, 
                kernel_initializer='normal', 
                activation='softmax'))



print(model.summary())





model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])




train_history =model.fit(x=x_Train_normalize,
                         y=y_Train_OneHot,validation_split=0.2, 
                         epochs=10, batch_size=32,verbose=1)



import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
	
	
	
	
show_train_history(train_history,'acc','val_acc')



show_train_history(train_history,'loss','val_loss')





scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])

