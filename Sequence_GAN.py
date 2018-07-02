import numpy as np
from keras.layers import Dense, Input, LeakyReLU, Flatten
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
from keras.models import Model
import time

def G():
    model = Sequential()

    # (NumberOfSamples, TimeSteps, Features)
    # define generator model architecture
    model.add(LSTM(1, batch_input_shape=(1,1,1),return_sequences=True))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(5,return_sequences=True,dropout=0.5))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(2,dropout=0.5))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1,activation='tanh'))
    model.add(Reshape((1,1)))
    model.summary()

    noise = Input(shape=[1,1,1])
    ans = model(noise)

    return Model(noise,ans)



def D():
    model = Sequential()
    # define discriminator model architecture
    model.add(LSTM(1, batch_input_shape=(1,1,1),return_sequences=True))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(LSTM(5,return_sequences=True))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(LSTM(2))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    value = Input(shape=[1,1,1])
    prob = model(value)

    return Model(value,prob)


def main():
    seq_1 = [-1,1,1,1]
    seq_2 = [1,-1,-1,-1]

    data = np.random.choice([seq_1,seq_2],size=100)




if __name__ == '__main__':
    main()