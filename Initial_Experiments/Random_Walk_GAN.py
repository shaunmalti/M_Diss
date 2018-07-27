import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LSTM
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
import keras.models as km
from keras.initializers import RandomNormal
from random import randint, uniform
from keras.callbacks import EarlyStopping
import csv
from sklearn.preprocessing import MinMaxScaler, normalize
import sys
import datetime
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
    model.add(LSTM(1, batch_input_shape=(10,1,1),return_sequences=True))
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

    return model


def G():
    model = Sequential()

    # (NumberOfSamples, TimeSteps, Features)
    # define generator model architecture
    model.add(LSTM(1, batch_input_shape=(10,1,1),return_sequences=True))
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

    return model


def train(data,disc_model,gen_model,combined,scalar,batch_size=10,train_steps=1000):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    # valid = np.random.normal(0.7,1.2,size=[batch_size,1])
    fake = np.zeros((batch_size, 1))

    # for w in range(0,10):
    x = 0
    for i in range(train_steps):
        np.random.seed(1)
        noise = np.random.normal(-1,1,size=[batch_size,1,1])
        # TODO this should be replaced by gen_model.predict()
        # gen_data = gen_model(tf.convert_to_tensor(noise,tf.float32))
        gen_data = gen_model.predict(noise)


        start_num = int(np.random.uniform(0, 1) * len(data))
        if start_num > (len(data) - batch_size):
            start_num -= batch_size
        data_ten = get_next_ten(data, start_num, batch_size)

        data_ten = np.reshape(data_ten, (10, 1, 1))

        # train discriminator
        d_loss_real = disc_model.train_on_batch(data_ten, valid)
        d_loss_fake = disc_model.train_on_batch(gen_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # reset discriminator state due to lstm
        disc_model.reset_states()

        # train generator
        np.random.seed(1)
        noise = np.random.normal(-1, 1, (10, 1, 1))
        # TODO should this be here?
        # gen_data_g = gen_model(tf.convert_to_tensor(noise, dtype=tf.float32))

        g_loss = combined.train_on_batch(noise, valid)

        # reset generator state due to lstm
        gen_model.reset_states()

        # TODO - is this wrong?
        print(i, ' [D loss:', d_loss, '] [G loss:', g_loss, ']')
        # print(data_ten)
        # print(gen_model.layers)
        # print(gen_model.layers[1].get_weights())
        # print(gen_model.layers[2].get_weights()[0])
        # print(gen_model.layers[3].get_weights()[0])
        x += 1



def main():
    # setting random seed for reproductability
    np.random.seed(1)
    start_time = time.time()
    data = import_csv() # imported data is stationary
    data_orig = data.copy()
    data = data[0:9000]
    # do it for the first 1000 - then perform test prediction
    # split data
    tf.reset_default_graph()
    optimiser = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=0.5)
    # optimiser = 'rmsprop'
    K.set_learning_phase(1)

    # perform normalisation
    # scalar = MinMaxScaler(feature_range=(0,1))
    # data = scalar.fit_transform(data)
    data_series = pd.DataFrame(data)
    scalar = MinMaxScaler(feature_range=(-1,1))
    data = scalar.fit_transform(data_series)

    data = np.arange(0,-5000,-1,dtype=float)
    scaler = MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data.reshape(-1,1))
    # TODO scalar.inverse_transform(data) at the end to get the data

    # plt.plot(scalar.inverse_transform(gen_model.predict(noise).reshape(-1, 1)))

    # build generator
    z = Input(shape=(1,1,),dtype=tf.float32)
    gen_model = G()
    # TODO check the commented line - should it be here?
    gen_model.compile(loss='mse',optimizer=optimiser,metrics=['accuracy'])
    output_gen = gen_model(z)

    # build discriminator
    disc_model = D()
    # only train generator
    # TODO - the .trainable False was removed - leads to better loss values
    disc_model.trainable = False
    # discriminator takes output from generator and determines probability
    prob_disc = disc_model(output_gen)
    # compile discriminator
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    disc_model.compile(loss='mse',optimizer=sgd,metrics=['accuracy'])

    # combined model
    # TODO commented to try other method
    # combined = Model(z, prob_disc)

    combined = Sequential()
    combined.add(gen_model)
    combined.add(disc_model)
    combined.compile(loss='mse',optimizer=optimiser,metrics=['accuracy'])

    # train model
    # TODO commented to try other method
    # train(data, gen_model, disc_model, combined, batch_size=1)

    train(data,disc_model,gen_model,combined,scalar)

    # perform one test maybe
    np.random.seed(1)
    noise = np.random.normal(-1, 1, (10, 1, 1))
    # gen_data_d = gen_model(tf.convert_to_tensor(noise, dtype=tf.float32))
    # print(gen_data_d)
    # to_predict = data[490:500].copy()
    # to_predict = np.reshape(to_predict, (10, 1, 1))
    # to_predict = np.float32(to_predict)
    # answer = gen_model.predict(to_predict)
    # answer_r = answer.copy().reshape((10,1))
    # un_normed_answer = scalar.inverse_transform(answer_r)
    # answer_n = gen_model.predict(noise)
    # answer_rn = answer_n.copy().reshape((10, 1))
    # un_normed_answer_n = scalar.inverse_transform(answer_rn)
    plt.plot(gen_model.predict(noise).reshape(-1, 1))
    # print('******DATA USED******')
    # print('DATA: ',scalar.inverse_transform(data[490:500]))
    # print('******ANSWERS WITH DATA BEING USED AS INPUT INTO GENERATOR******')
    # print('UNNORMALISED ANSWER: ',un_normed_answer[0])
    # print('NORMALISED ANSWER: ',answer_r[0])
    # print('DISCRIMINATOR PROB. VALUE: ',disc_model.predict(answer)[0])
    # print('******ANSWERS WITH NOISE BEING USED AS INPUT INTO GENERATOR******')
    # print('UNNORMALISED ANSWER: ',un_normed_answer_n[0])
    # print('NORMALISED ANSWER: ',answer_rn[0])
    # print('DISCRIMINATOR PROB. VALUE: ',disc_model.predict(answer_n)[0])
    # print('******ACTUAL ANSWER******')
    # print('UNNORMALISED: ',scalar.inverse_transform(data[500].reshape(1,-1)))
    # print('NORMALISED: ', data[500])
    # print('Time taken: ', time.time()-start_time)
    # print('******TOTAL GEN ANSWER******')
    # print('NOISE ANSWER UNNORMED')
    # print(un_normed_answer_n)
    # print('INPUT ANSWER UNNORMED')
    # print(un_normed_answer)
    # print('MSE w/ data: ',((un_normed_answer - data[500:510]) ** 2).mean(axis=0))
    # print('MSE w/ noise: ',((un_normed_answer_n - data[500:510]) ** 2).mean(axis=0))
    # plt.plot(un_normed_answer_n,color='green',label='Unnormed Answer Noise')
    # plt.plot(un_normed_answer, color='red',label='Unnormed Answer Data')
    # plt.plot(scalar.inverse_transform(data[500:510]),color='blue',label='Actual Answer')
    # plt.legend()
    plt.show()

def get_next_ten(all_data,start_num,batch_size):
    return all_data[start_num:(start_num+batch_size)]


if __name__ == '__main__':
    main()