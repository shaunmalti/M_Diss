# Making a simple GAN which maps distribution of -1, 1 choices
# this should be a 50/50 split
import numpy as np
from keras.layers import Dense, Input, LeakyReLU, Flatten
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
from keras.models import Model
import time

def G():
    model = Sequential()

    model.add(Dense(1,input_dim=1,activation=None))
    # model.add(Dense(1,activation=None))
    model.add(Dense(1,activation=None))
    model.summary()

    noise = Input(shape=[1])
    ans = model(noise)

    return Model(noise,ans)



def D():
    model = Sequential()

    model.add(Dense(1,input_dim=1))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(2))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(3))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    value = Input(shape=[1])
    prob = model(value)

    return Model(value,prob)



def train(gen_model,disc_model,combined,data):
    valid = np.ones((1,1))
    fake = np.ones((1,1))
    d_loss_arr = []
    g_loss_arr = []
    for i in range(0,len(data)):

        # define batches
        start_num = int(np.random.uniform(0,1)*len(data))
        if start_num > (len(data)-10):
            start_num -= 10

        # pick batches

        # TRAIN DISCRIMINATOR
        np.random.seed(1)
        # noise = np.random.normal(-1,1,size=[1])
        noise = np.random.choice([-1,1],size=[1])

        gen_data = gen_model.predict(noise)

        d_loss_real = disc_model.train_on_batch(np.reshape(data[i],[1,1]),valid)
        d_loss_fake = disc_model.train_on_batch(gen_data,fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_loss_arr.append(d_loss)
        # TRAIN GENERATOR

        # TO BE REMOVED
        disc_model.trainable = False
        # temptest = disc_model.predict(gen_data)
        # temptest2 = combined.predict(noise)
        # TODO TO BE REMOVED

        np.random.seed(1)
        # noise_2 = np.random.normal(-1,1,size=[1])
        noise_2 = np.random.choice([1,-1],size=[1])

        g_loss = combined.train_on_batch(noise_2,valid)
        g_loss_arr.append(g_loss)
        print(i, ' [D loss:', d_loss, '] [G loss:', g_loss, ']')
        print(disc_model.predict(gen_data))

    plt.plot(d_loss_arr,color='r')
    plt.plot(g_loss_arr,color='b')
    plt.show()


def main():
    # setting random seed for reproducable values
    np.random.seed(1)
    data = np.random.choice([-1,1],size=500)
    unique,counts = np.unique(data,return_counts=True)
    data_dict = dict(zip(unique,counts))
    # TODO - best till now with this Adam
    # optimiser = Adam(lr=0.002,beta_1=0.8,beta_2=1)
    optimiser = Adam(lr=0.002, beta_1=0.8, beta_2=1)
    disc_optimiser = SGD(lr=0.01)

    disc_model = D()
    disc_model.compile(loss='mean_squared_error', optimizer=optimiser)

    gen_model = G()

    z = Input(shape=[1])
    output_gen = gen_model(z)

    disc_model.trainable = False

    prob = disc_model(output_gen)

    combined = Model(z,prob)
    combined.compile(loss='mean_squared_error', optimizer=optimiser)


    train(gen_model,disc_model,combined,data)

    list_ans = []
    actual_ans = []
    # np.random.seed(1)
    np.random.seed(seed=int(time.time()))
    for i in range(0,50):
        # noise_t = np.random.normal(-1,1,size=[1])
        noise_t = np.random.choice([1,-1],size=[1])
        ans = gen_model.predict(noise_t)
        print(ans)
        actual_ans.append(ans)
        if ans > 0:
            list_ans.append(1)
        else:
            list_ans.append(-1)

    unique,counts = np.unique(list_ans,return_counts=True)
    ans_dict = dict(zip(unique,counts))
    # print(actual_ans)
    print('Original Distribution: ', data_dict)
    print('Generated Distribution: ', ans_dict)

    # noise_t = np.random.normal(-1, 1, size=[1])
    noise_t = np.random.choice([1, -1], size=[1])
    ans = gen_model.predict(noise_t)
    print('Noise Value: ', noise_t)
    print('Predicted Ans, ',ans)
    print('Disc Output, ',disc_model.predict(ans))
    print('Combined output, ',combined.predict(noise_t))

    print('Separate Test')
    print('Inputting Random Value')
    np.random.seed(seed=int(time.time()))
    # print(gen_model.predict([[np.random.normal(-1, 1, size=[1])]]))
    print(gen_model.predict([[np.random.choice([-1, 1], size=[1])]]))
    # print('Inputting Random Value')
    # print(gen_model.predict([[np.random.normal(-1, 1, size=[1])]]))

    # plt.plot(actual_ans)
    # plt.show()


if __name__ == '__main__':
    main()