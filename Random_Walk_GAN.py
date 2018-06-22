import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LSTM
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.initializers import RandomNormal
from random import randint, uniform
import csv
from sklearn.preprocessing import normalize

def import_csv():
    with open('rand_walk.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        data = []
        for row in readCSV:
            data.append(float(row[0]))
    return data


def D():
    model = Sequential()

    # define discriminator model architecture
    model.add(LSTM(1, batch_input_shape=(1,1,1),stateful=True,kernel_initializer='random_uniform',
                   bias_initializer='zeros',return_sequences=True))
    model.add(LeakyReLU(alpha=0.2))
    model.add(LSTM(5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    return model


def G():
    model = Sequential()

    # (NumberOfSamples, TimeSteps, Features)
    # define generator model architecture
    model.add(LSTM(1, batch_input_shape=(1,1,1),stateful=True,kernel_initializer='random_uniform',
                   bias_initializer='zeros',return_sequences=True))
    model.add(LeakyReLU(alpha=0.2))
    model.add(LSTM(5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1,activation='tanh'))
    model.add(Reshape((1,1)))
    model.summary()

    return model


def train(data, gen_model, disc_model, combined, batch_size=1, epochs=10):
    # adversarial ground truths
    valid = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    data = np.reshape(data,(data.shape[0],1,1))

    for epoch in range(epochs):
        # define noise to be fed into generator
        noise = np.random.normal(0, 1, (1,1,1))

       # generated data
        gen_data = gen_model(tf.convert_to_tensor(noise,dtype=tf.float32))

        # train discriminator
        # TODO error in line under - lstm3 input expects 3 dimensions but  have array with shape 99999,1
        # TODO added line that does np.reshape outside for loop to fix this
        d_loss_real = disc_model.train_on_batch(data, valid)
        d_loss_fake = disc_model.train_on_batch()

        # reset discriminator state due to lstm
        disc_model.reset_states()

        # train generator
        noise = np.random.normal(0, 1, (batch_size, 1))
        g_loss = combined.train_on_batch(noise, valid)

        # reset generator state due to lstm
        gen_model.reset_states()


def main():
    data = import_csv() # imported data is stationary
    # might have to do reshaped_data = np.reshape(data,(-1,1)) later on

    optimiser = Adam(0.0002,0.5)

    # build generator
    z = Input(shape=(1, 1))
    gen_model = G()
    # TODO check the commented line - should it be here?
    # gen_model.compile(loss='mean_squared_logarithmic_error',optimizer=optimiser)
    output_gen = gen_model(z)

    # build discriminator
    disc_model = D()
    # only train generator
    disc_model.trainable = False
    # discriminator takes output from generator and determines probability
    prob_disc = disc_model(output_gen)
    # compile discriminator
    disc_model.compile(loss='mean_squared_logarithmic_error',optimizer=optimiser)

    # combined model
    combined = Model(z, prob_disc)
    combined.compile(loss='mean_squared_logarithmic_error',optimizer=optimiser)

    train(data, gen_model, disc_model, combined, batch_size=1, epochs=10)

if __name__ == '__main__':
    main()