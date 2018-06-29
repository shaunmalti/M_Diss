# Making a simple GAN which maps distribution of -1, 1 choices
# this should be a 50/50 split
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt

def G():
    model = Sequential()

    model.add(Dense(1,input_dim=1))
    model.add(Dense(2))
    model.add(Dense(1))
    model.summary()

    return model



def D():
    model = Sequential()

    model.add(Dense(1, input_dim=1))
    model.add(Dense(2))
    model.add(Dense(1))
    model.summary()

    return model



def main():
    # setting random seed for reproducable values
    np.random.seed(1)
    data = np.random.choice([-1,1],size=10000)
    unique,counts = np.unique(data,return_counts=True)
    data_dict = dict(zip(unique,counts))
    print(data)
    print(data_dict)
    # optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.5)
    optimiser = Adam()

    z = Input(shape=[1])

    gen_model = G()
    gen_model.compile(loss='mse', optimizer=optimiser, metrics=['accuracy'])
    # output_gen = gen_model(z)

    disc_model = D()

    disc_model.trainable = False

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    disc_model.compile(loss='mse',optimizer=optimiser,metrics=['accuracy'])

    combined = Sequential()
    combined.add(gen_model)
    combined.add(disc_model)
    combined.compile(loss='mse', optimizer=optimiser, metrics=['accuracy'])

    train(gen_model,disc_model,combined,data)

    list_ans = []
    for i in range(0,100):
        noise_t = np.random.normal(-1,1,size=[1])
        list_ans.append(np.reshape(gen_model.predict(noise_t),1))

    plt.plot(list_ans)
    plt.show()


def train(gen_model,disc_model,combined,data):
    valid = np.ones((1,1))
    fake = np.ones((1,1))

    for i in range(0,len(data)):
        np.random.seed(1)
        noise = np.random.normal(-1,1,size=[1])

        gen_data = gen_model.predict(noise)

        d_loss_real = disc_model.train_on_batch(np.reshape(data[i],[1,1]),valid)
        d_loss_fake = disc_model.train_on_batch(gen_data,fake)
        d_loss = 0.5 * np.add(d_loss_fake,d_loss_real)

        np.random.seed(1)
        noise_2 = np.random.normal(-1,1,size=[1])

        g_loss = combined.train_on_batch(noise_2,valid)

        print(i, ' [D loss:', d_loss, '] [G loss:', g_loss, ']')



if __name__ == '__main__':
    main()