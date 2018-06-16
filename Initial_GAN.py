import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LSTM
from keras.activations import relu
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.initializers import RandomNormal

def import_csv():
    # import, set header names and change columns depending on content
    data = pd.read_csv('FF_alt.txt',sep=',',names=['Date','Time','Open','High','Low','Close','Volume'])

    # pick relevant columns and redefine as numpy array
    target = data[['Close']].as_matrix()
    data = data[['Open','High','Low','Volume']].as_matrix()

    # perform difference of time_series from one step to the next
    # so as to make data stationary
    # differenced_data = differenced_series(data, interval=1)
    return data, target

def differenced_series(data, interval):
    # iterate over data line by line and perform difference with
    # previous lines value
    diff = [[0 for x in range(len(data[0]))] for w in range(len(data))]
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            diff[i][j] = data[i][j] - data[i - interval][j]
    return diff

def G():
    model = Sequential()

    # (NumberOfSamples, TimeSteps, Features)
    # define generator model architecture
    model.add(LSTM(4, batch_input_shape=(1,1,4),stateful=True,kernel_initializer='random_uniform',
                   bias_initializer='zeros',return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(10))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.summary()

    return model


def D():
    model = Sequential()

    # define discriminator model architecture
    model.add(LSTM(1, batch_input_shape=(1,1,1),stateful=True,kernel_initializer='random_uniform',
                   bias_initializer='zeros',return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(10))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.summary()

    return model



def main():
    data, target_data = import_csv()


    batch_size = 128

    # test_model = G(data.shape, batch_size)
    test_model = D()

    test_model.compile(loss='mean_squared_logarithmic_error',optimizer='adam')

    # (NumberOfSamples, TimeSteps, Features)
    X = data.reshape(len(data), 1, 4)
    print(X.shape)
    num_epochs = 10

    for i in range(num_epochs):
        test_model.fit(X,target_data,epochs=1,batch_size=1, shuffle=False)
        test_model.reset_states()
    # need to make the data stationary

    x = np.array([99.825, 99.825, 99.825, 1])
    x = x.reshape(1, 1, 4)
    print(test_model.predict(x))

    # G()
    test = 123




if __name__ == '__main__':
    main()