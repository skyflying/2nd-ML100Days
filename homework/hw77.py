import os
import keras

# ���@�~�i�H���ݨϥ� GPU, �N GPU �]�w�� "�L" (�Y�� GPU �B�Q�}�ҡA�i�]�� "0")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# �q Keras �����إ\�त�A���o train �P test ��ƶ�
train, test = keras.datasets.cifar10.load_data()



# �N X �P Y �W�ߩ�i�ܼ�
x_train, y_train = train
x_test, y_test = test
# ��ƫe�B�z - �зǤ�
x_train = x_train / 255.
x_test = x_test / 255.

# �N��Ʊq�ϧ� (RGB) �ର�V�q (Single Vector)
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# �N�ؼ��ର one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)





def build_mlp():
    input_layer = keras.layers.Input([x_train.shape[-1]])
    x = keras.layers.Dense(units=512, activation="relu")(input_layer)
    x = keras.layers.Dense(units=256, activation="relu")(x)
    x = keras.layers.Dense(units=128, activation="relu")(x)
    out = keras.layers.Dense(units=10, activation="softmax")(x)
    
    model = keras.models.Model(inputs=[input_layer], outputs=[out])

    return model
model = build_mlp()





# �� Keras ���ؤ�k�˵��ҫ��U�h�Ѽƶq
model.summary()

optimizer = keras.optimizers.SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)




model.fit(x_train, y_train, 
          epochs=500, 
          batch_size=256, 
          validation_data=(x_test, y_test), 
          shuffle=True)
		  
		  
		  
		  
		  
		  
import matplotlib.pyplot as plt
# �H��ı�e�覡�˵��V�m�L�{

train_loss = model.history.history["loss"]
valid_loss = model.history.history["val_loss"]

train_acc = model.history.history["acc"]
valid_acc = model.history.history["val_acc"]

plt.plot(range(len(train_loss)), train_loss, label="train loss")
plt.plot(range(len(valid_loss)), valid_loss, label="valid loss")
plt.legend()
plt.title("Loss")
plt.show()

plt.plot(range(len(train_acc)), train_acc, label="train accuracy")
plt.plot(range(len(valid_acc)), valid_acc, label="valid accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()