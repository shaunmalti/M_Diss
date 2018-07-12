import numpy as np
from keras.layers import Dense, Input, LeakyReLU, Flatten, LSTM, Reshape, ReLU, Activation
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
from keras.models import Model
from keras.initializers import Constant,Ones,Zeros
import time
import tensorflow as tf

def G():
    model = Sequential()
    # TODO check effect of dropout, first without

    # (NumberOfSamples, TimeSteps, Features)
    # define generator model architecture
    model.add(LSTM(1, batch_input_shape=(10,1,1),return_sequences=True,activation=None))
    # model.add(Activation('relu'))
    model.add(LSTM(2,return_sequences=True,activation=None)) # kernel_initializer=Constant(value=10)
    # model.add(LSTM(5, activation=None))
    # model.add(Activation('relu'))
    model.add(Dense(1,activation=None))
    # model.add(Activation('relu'))
    model.add(Reshape((1, 1)))
    model.summary()

    noise = Input(shape=[1,1,])
    ans = model(noise)

    return Model(noise,ans)
    # return model


def D():
    model = Sequential()
    # define discriminator model architecture
    model.add(LSTM(1, batch_input_shape=(10,1,1),return_sequences=True, activation=None))

    # model.add(Dense(1,input_shape=(10,1,1),activation=None))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    # model.add(LSTM(2,return_sequences=True, activation=None))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    # model.add(LSTM(5,activation=None))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    value = Input(shape=[1,1,])
    prob = model(value)

    return Model(value,prob)
    # return model

def get_next_ten(all_data,start_num):
    return all_data[start_num:(start_num+10)]

def train(all_data, gen_model, disc_model, combined):
    valid = np.ones(shape=[10,1,1])
    fake = np.zeros(shape=[10,1,1])

    num_epochs = 100000

    d_loss_arr = []
    g_loss_arr = []
    for i in range(num_epochs):
        # define batches
        start_num = int(np.random.uniform(0,1)*len(all_data))
        if start_num > (len(all_data)-10):
            start_num -= 10

        train_data = get_next_ten(all_data, start_num)

        # TRAIN DISCRIMINATOR
        # np.random.seed(1)
        # noise = np.random.choice([-1, 1], size=[10,1,1])
        noise = np.random.normal(0,size=[10,1,1])

        gen_data = gen_model.predict(noise)

        d_loss_real = disc_model.train_on_batch(np.reshape(train_data,[10,1,1]),valid)
        d_loss_fake = disc_model.train_on_batch(gen_data,fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_loss_arr.append(d_loss)
        # print(i, disc_model.predict(gen_data))


        # TRAIN GENERATOR

        # define batches
        start_num = int(np.random.uniform(0,1)*len(all_data))
        if start_num > (len(all_data)-10):
            start_num -= 10

        train_data = get_next_ten(all_data, start_num)
        # for w in range(0,10):
        #
        # # np.random.seed(1)
        # # noise_2 = np.random.choice([1, -1], size=[10,1,1])
        #     noise_2 = np.random.normal(0,size=[10,1,1])
        #
        #     g_loss = combined.train_on_batch(noise_2, valid)
        #     g_loss_arr.append(g_loss)
        noise_2 = np.random.choice([1, -1], size=[10, 1, 1])

        g_loss = combined.train_on_batch(noise_2, valid)
        g_loss_arr.append(g_loss)


        print(i, ' [D loss:', d_loss, '] [G loss:', g_loss, ']')
        # print(disc_model.predict(gen_data))
    plt.title('G and D Losses with 10x G training')
    plt.plot(d_loss_arr,color='r',label='D')
    plt.plot(g_loss_arr,color='b',label='G')
    plt.legend()
    plt.show()

def main():

    # DEFINE SEQUENCES OF DATA
    seq_1 = [-1,-1,-1,-1]
    seq_2 = [1000,1000,1000,1000]

    # tf.reset_default_graph()
    # np.random.seed(1)
    data = np.random.choice([1,2],size=1000)
    # IMPORTANT - given a test of just repetitions of -1,1,1,1. Would the GAN create long runs of positive
    # numbers with just one negative
    # data = np.random.choice([1],size=100)

    unique,counts = np.unique(data,return_counts=True)
    data_dict = dict(zip(unique,counts))
    all_data = []
    for i in range(len(data)):
        if data[i] == 1:
            all_data.append(seq_1)
        else:
            all_data.append(seq_2)

    all_data = np.ndarray.flatten(np.array(all_data))

    # DEFINE G AND D

    # define optimisers
    # optimiser = Adam(lr=0.01,decay=1e-6)
    optimiser = Adam()
    optimiser_g = Adam()
    # sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9,nesterov=True)

    # define discriminator and compile
    disc_model = D()
    disc_model.compile(loss='mean_squared_error', optimizer=optimiser)

    # define generator and adversarial model, compile
    gen_model = G()

    # behaviour
    z = Input(shape=[1,1,])
    output_gen = gen_model(z)

    disc_model.trainable = False

    prob = disc_model(output_gen)

    # adversarial model
    combined = Model(z,prob)
    combined.compile(loss='mean_squared_error', optimizer=optimiser_g)

    # TODO REMOVE FROM HERE
    test_ar = []
    combined_in_arr = []
    for i in range(0,100):
        noise_t = np.random.normal(0, size=[10, 1, 1])
        # noise_t = np.random.choice([-1, 1], size=[10, 1, 1])
        ans = gen_model.predict(noise_t)
        ans_c = combined.predict(noise_t)
        test_ar.append(np.reshape(ans,(10,1)))

        combined_in_arr.append(np.reshape(ans_c,(10,1)))
    print(np.reshape(ans,(10,1)))
    print(np.reshape(disc_model.predict(ans),(10,1)))
    print(data_dict)
    test_ar_flat = np.ndarray.flatten(np.array(test_ar))
    comb_ar_flat = np.ndarray.flatten(np.array(combined_in_arr))
    plt.title('Outputs Throughout BEFORE TRAIN')
    plt.plot(test_ar_flat)
    plt.plot(comb_ar_flat)
    plt.show()

    for i in range(1,len(test_ar_flat)):
        test_ar_flat[i] = test_ar_flat[i] + test_ar_flat[i - 1]
    test_ar_flat = np.delete(test_ar_flat, 0)
    plt.title('Prediction for after BEFORE TRAIN')
    plt.plot(test_ar_flat)
    plt.show()
    # TODO REMOVE TILL HERE

    # perform training
    train(all_data, gen_model, disc_model, combined)


    # perform single 10 stage test
    # np.random.seed(1)
    test_ar = []
    combined_in_arr = []
    for i in range(0,100):
        noise_t = np.random.normal(0, size=[10, 1, 1])
        # noise_t = np.random.choice([-1, 1], size=[10, 1, 1])
        ans = gen_model.predict(noise_t)
        ans_c = combined.predict(noise_t)
        test_ar.append(np.reshape(ans,(10,1)))

        combined_in_arr.append(np.reshape(ans_c,(10,1)))

        # TODO need to perform rolling train to check effect on generated data

        # valid = np.ones(shape=[10, 1, 1])
        # fake = np.zeros(shape=[10, 1, 1])
        #
        # # TRAIN DISCRIMINATOR
        # # np.random.seed(1)
        # noise = np.random.choice([-1, 1], size=[10, 1, 1])
        #
        # gen_data = gen_model.predict(noise)
        #
        # disc_model.train_on_batch(ans, fake)
        # # TRAIN GENERATOR
        #
        # noise_2 = np.random.choice([1, -1], size=[10, 1, 1])
        #
        # g_loss = combined.train_on_batch(noise_2, valid)


    print(np.reshape(ans,(10,1)))
    print(np.reshape(disc_model.predict(ans),(10,1)))
    print(data_dict)
    test_ar_flat = np.ndarray.flatten(np.array(test_ar))
    comb_ar_flat = np.ndarray.flatten(np.array(combined_in_arr))
    plt.title('Outputs Throughout')
    plt.plot(test_ar_flat)
    plt.plot(comb_ar_flat)
    plt.show()

    for i in range(1,len(test_ar_flat)):
        test_ar_flat[i] = test_ar_flat[i] + test_ar_flat[i - 1]
    test_ar_flat = np.delete(test_ar_flat, 0)
    plt.title('Prediction for after')
    plt.plot(test_ar_flat)
    plt.show()

if __name__ == '__main__':
    main()