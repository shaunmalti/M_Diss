import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib import rnn
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import random

def import_csv(window_size):
    # import, set header names and change columns depending on content
    data = pd.read_csv('C.txt',sep=',',names=['Date','Open','High','Low','Close','Volume'])

    # redefine data as being 0/1. 1 when close > open and 0 in all other cases
    data_np = data.values
    new_array = []

    for i in range(0,len(data_np)):
        if data_np[i][4] > data_np[i][1]:
            new_array.append(1)
        else:
            new_array.append(-1)

    for i in range(1, len(new_array)):
        new_array[i] = new_array[i] + new_array[i - 1]

    print(hurst(new_array))
    series = pd.DataFrame(new_array)
    series_s = series.copy()
    for i in range(1,window_size):
        series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)

    series.dropna(axis=0, inplace=True)

    return series.values

def hurst(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m = polyfit(log(lags), log(tau), 1)
    return m[0] * 2.0

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def create_cell(size):
    lstm_cell = rnn.LSTMCell(size,forget_bias=0.7, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(lstm_cell)
    return lstm_cell

def generator(Z,batch_size,reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        # Z = tf.unstack(Z,axis=2)
        cell = rnn.MultiRNNCell([create_cell(2),create_cell(10),create_cell(30)],state_is_tuple=True)
        outputs, states = rnn.static_rnn(cell, [Z], dtype=tf.float32,scope="GAN/Generator")
        # test = tf.reshape(outputs[-1], shape=[None, outputs[-1].shape[1], 1])
    return outputs[-1]

def lstm_cell(single=True):
    if single is True:
        return rnn.LSTMCell(128,forget_bias=0.7, state_is_tuple=True)
    else:
        return rnn.LSTMCell(1,forget_bias=0.7, state_is_tuple=True)

def discriminator(X,batch_size,reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.LSTMCell(30,forget_bias=0.7, state_is_tuple=True),
             rnn.LSTMCell(10, forget_bias=0.7, state_is_tuple=True),
             rnn.LSTMCell(1, forget_bias=0.7, state_is_tuple=True)], state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(stacked_lstm, X, dtype=tf.float32)
        out = tf.nn.sigmoid(outputs[-1])
    return out

def get_next(all_data,start_num,batch_size,window_size):
    return np.reshape(all_data[start_num:(start_num+batch_size)],newshape=[batch_size,window_size,1])

def shuffle_x(X_batch):
    batch_indices = list(range(len(X_batch)))
    random.shuffle(batch_indices)
    new_array = []
    for i in range(len(batch_indices)):
        new_array.append(X_batch[batch_indices[i]])

    return np.array(new_array)


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    """need to switch input to 3d [?,30,1] with the 30 being the next 30 steps/readings. Therefore
    removing the x argument currently there - i.e. [0,w]...[len(data),w]"""
    window_size = 30
    data = import_csv(window_size)

    tf.reset_default_graph()
    # data_x = np.arange(0,len(data_y),1,dtype=float).reshape(len(data_y),1)
    # scalar_x = MinMaxScaler(feature_range=(-1,1))
    # scalar_x.fit(np.array(data_x).reshape(-1, 1))
    # scalar_y = MinMaxScaler(feature_range=(-1,1))
    # scalar_y.fit(np.array(data_y).reshape(-1, 1))
    # data_y = np.array(data_y).reshape(-1,1)
    # data_x = np.array(data_x).reshape(-1,1)

    batch_size = 64
    # data = data_x
    # data = np.concatenate([data_x, data_y], axis=1)

    X = tf.placeholder(tf.float32, [None, window_size,1])
    Z = tf.placeholder(tf.float32, [None, 1])

    G_sample = generator(Z, batch_size)
    r_logits = discriminator(X, batch_size)
    G_sample = tf.expand_dims(G_sample, dim=2)
    f_logits = discriminator(G_sample, batch_size, reuse=True)

    # gen_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(f_logits),predictions=f_logits))
    #
    # d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(r_logits),predictions=r_logits))
    # d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(f_logits),predictions=f_logits))
    # disc_loss = d_loss_fake + d_loss_real

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)) +
                               tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

    # gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
    # disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step
    gen_step = tf.train.AdamOptimizer().minimize(gen_loss, var_list=gen_vars)
    disc_step = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disc_vars)

    # sess = tf.Session(config=config)
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    nd_steps = 10
    ng_steps = 10

    gen_loss_arr = []
    disc_loss_arr = []

    for i in range(500):
        start_num = int(np.random.uniform(0, 1) * len(data))
        if start_num > (len(data) - batch_size):
            start_num -= batch_size
        # if i == 0:

        X_batch = get_next(data,start_num,batch_size,window_size)
        X_batch = shuffle_x(X_batch)
        Z_batch = sample_Z(batch_size, 1)

        # for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

        # for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})


        # print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
        gen_loss_arr.append(gloss)
        disc_loss_arr.append(dloss)
        if i % 10 == 0:
            print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, dloss, gloss))
        # if i % 100 == 0:
            # answer = sess.run(G_sample, feed_dict={Z: Z_batch})
            # ans_y = answer[:, 1]
            # ans_x = answer[:, 0]
            # ans_y_re = scalar_x.inverse_transform(ans_y.reshape(-1, 1))
            # ans_x_re = scalar_y.inverse_transform(ans_x.reshape(-1, 1))
            # # plt.scatter(ans_x,ans_y)
            # plt.scatter(ans_x_re, ans_y_re)
            # # plt.plot(np.reshape(answer,newshape=(batch_size,1)),color='r',label='Pred')
            # # plt.plot(np.reshape(x_batch,newshape=(batch_size,1)),color='b',label='Actual')
            # plt.title('Prediction Iter: {0}'.format(i))
            # plt.savefig('./iteration_pred_{0}'.format(i))
            # plt.close()

    plt.plot(disc_loss_arr, color='r', label='D')
    plt.plot(gen_loss_arr, color='b', label='G')
    plt.legend()
    plt.show()

    test_batch = sample_Z(1, 1)

    answer = sess.run(G_sample, feed_dict={Z: test_batch})
    # line = np.arange(0,len(answer),1)
    # ans_y = answer[:, 1]
    # ans_x = answer[:, 0]
    # ans_y_re = scalar_y.inverse_transform(ans_y.reshape(-1, 1))
    # ans_x_re = scalar_x.inverse_transform(ans_x.reshape(-1, 1))
    plt.plot(answer[0])
    plt.show()
    # plt.scatter(ans_x_re,ans_y_re)
    print(hurst(answer[0]))
    plt.show()

    test = tf.train.Saver()


if __name__ == '__main__':
    main()

    """THEREFORE ADDING THE SECOND DIMENSION - X AXIS - LEADS TO A GOOD RESULT"""