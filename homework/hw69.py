from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import warnings
warnings.filterwarnings("ignore")



#�D�n��J�����s�D���D�����A�Y�@�Ӿ�ƧǦC�]�C�Ӿ�ƽs�X�@�ӵ��^�C
#�o�Ǿ�Ʀb1 ��10,000 �����]10,000 �ӵ������J��^�A�B�ǦC���׬�100 �ӵ�
#�ŧi�@�� NAME �h�w�qInput
main_input = Input(shape=(100,), dtype='int32', name='main_input')


# Embedding �h�N��J�ǦC�s�X���@�ӸY�K�V�q���ǦC�A
# �C�ӦV�q���׬� 512�C
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM �h��V�q�ǦC�ഫ����ӦV�q�A
# ���]�t��ӧǦC���W�U��H��
lstm_out = LSTM(32)(x)






#���J���U�l���A�ϱo�Y�Ϧb�ҫ��D�l���ܰ������p�U�ALSTM �h�MEmbedding �h����Q��í�a�V�m
news_output = Dense(1, activation='sigmoid', name='news_out')(lstm_out)




#���U��J�ƾڻPLSTM �h����X�s���_�ӡA��J��ҫ�

news_input = Input(shape=(5,), name='news_in')
x = keras.layers.concatenate([lstm_out, news_input])


# ���|�h�ӥ��s�������h
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
#�@�~�ѵ�: �s�W��h
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# �̫�K�[�D�n���޿�^�k�h
main_output = Dense(1, activation='sigmoid', name='main_output')(x)


model = Model(inputs=[main_input, news_input], outputs=[main_output, news_output])

model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'news_out': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'news_out': 0.2})



model.summary()

