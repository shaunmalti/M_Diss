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
import sys
from IPython.display import HTML, display, clear_output
import time
import math

tf.reset_default_graph()
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

def import_csv(filename, dir):
    # import, set header names and change columns depending on content
    data = pd.read_csv(dir + filename,sep=',',names=['Date','Open','High','Low','Close','Volume'])

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

    return np.array(new_array)

def add_timesteps(data, window_size):
    #     scalar = MinMaxScaler(feature_range=(-1,1))
    #     scalar = scalar.fit(np.array(new_array).reshape(-1,1))
    #     new_array = scalar.transform(np.array(new_array).reshape(-1,1))
    series = pd.DataFrame(data)
    series_s = series.copy()
    for i in range(0, window_size):
        series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
    series.dropna(axis=0, inplace=True)
    #     return series.values, scalar
    arr = np.delete(series.values, -1, axis=1)
    return arr


def hurst(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m = polyfit(log(lags), log(tau), 1)
    return m[0] * 2.0


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def create_cell(size):
    lstm_cell = rnn.LSTMCell(size, state_is_tuple=True, activation=tf.nn.leaky_relu)
    lstm_cell = rnn.DropoutWrapper(lstm_cell)
    return lstm_cell


# removed dropout wrapper
# removed forget bias from both g and d
# make no of timesteps 128

def generator(Z, batch_size, reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        # cell = rnn.MultiRNNCell([create_cell(2), create_cell(5), create_cell(64)], state_is_tuple=True)
        # outputs, states = tf.nn.dynamic_rnn(cell, Z, dtype=tf.float32, scope="GAN/Generator")
        # out = tf.nn.tanh(outputs[-1])

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(1), rnn.BasicLSTMCell(10), rnn.BasicLSTMCell(64)])
        outputs, states = rnn.static_rnn(rnn_cell, [Z], dtype=tf.float32)
        output = tf.nn.tanh(outputs[-1])
    return output


def lstm_cell(single=True):
    if single is True:
        return rnn.LSTMCell(128, state_is_tuple=True, activation=tf.nn.leaky_relu)
    else:
        return rnn.LSTMCell(1, state_is_tuple=True, activation=tf.nn.leaky_relu)


def discriminator(X, batch_size, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        # stacked_lstm = rnn.MultiRNNCell(
        #     [rnn.LSTMCell(64, state_is_tuple=True),
        #      rnn.LSTMCell(5, state_is_tuple=True),
        #      rnn.LSTMCell(1, state_is_tuple=True)], state_is_tuple=True)
        # outputs, states = tf.nn.dynamic_rnn(stacked_lstm, X, dtype=tf.float32)
        # out = tf.layers.dense(inputs=outputs[-1], units=1, activation=tf.nn.sigmoid)

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(64), rnn.BasicLSTMCell(2),rnn.BasicLSTMCell(1)])
        outputs, states = rnn.static_rnn(rnn_cell, [X], dtype=tf.float32,scope='stat_g')
        output = tf.nn.sigmoid(outputs[-1])

    # return out, outputs[-1]
    return output


def get_next(sin_num, all_data, start_num, batch_size, window_size):
    # return np.reshape(all_data[sin_num][start_num:(start_num + batch_size)], newshape=[batch_size, window_size])
    return np.reshape(all_data[0][0:(0 + batch_size)], newshape=[batch_size, window_size])

def shuffle_x(X_batch):
    batch_indices = list(range(len(X_batch)))
    random.shuffle(batch_indices)
    new_array = []
    for i in range(len(batch_indices)):
        new_array.append(X_batch[batch_indices[i]])

    return np.array(new_array)


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def sampled_sine_wave(freq):
    samples = 1000
    time_period = 500
    time = np.linspace(0,time_period,samples)
    Vin1=([np.sin(t*freq*2*np.pi) for t in time])
    return Vin1


def main():
    sin1 = np.sin(np.linspace(0, 100, num=1000))
    sin2 = np.sin(np.linspace(0, 200, num=1000))
    sin3 = np.sin(np.linspace(0, 300, num=1000))

    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    window_size = 64
    data = []
    sin1 = add_timesteps(sin1, window_size)
    sin2 = add_timesteps(sin2, window_size)
    sin3 = add_timesteps(sin3, window_size)

    data.append(sin1)
    data.append(sin2)
    data.append(sin3)

    batch_size = 128

    X = tf.placeholder(tf.float32, [None, window_size])
    Z = tf.placeholder(tf.float32, [None, 1])

    G_sample = generator(Z, batch_size)

    d_real = discriminator(X, batch_size)
    # G_sample = tf.expand_dims(G_sample, dim=2)
    G_sample = tf.reshape(G_sample,shape=[batch_size,window_size])
    d_fake = discriminator(G_sample, batch_size, reuse=True)

    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)), 1)
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)), 1)
    disc_loss = disc_loss_fake + disc_loss_real

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)), 1)

    # gen_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(f_logits),predictions=f_logits))

    # d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(r_logits),predictions=r_logits))
    # d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(f_logits),predictions=f_logits))
    # disc_loss = d_loss_fake + d_loss_real


    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

    gen_step = tf.train.AdamOptimizer(0.02).minimize(gen_loss, var_list=gen_vars)
    disc_step = tf.train.AdamOptimizer(0.001).minimize(disc_loss, var_list=disc_vars)
    # gen_step = tf.train.GradientDescentOptimizer(0.002).minimize(gen_loss,var_list=gen_vars)
    # disc_step = tf.train.GradientDescentOptimizer(0.001).minimize(disc_loss,var_list=disc_vars)

    nd_steps = 1
    ng_steps = 10

    gen_loss_arr = []
    disc_loss_arr = []

    iterations = 2000
    # out = display(progress(0, iterations), display_id=True)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # tf.global_variables_initializer().run(session=sess)
    sess.run(tf.global_variables_initializer())

    start_time_gpu = time.time()
    for i in range(iterations):

        sin_num = np.random.randint(low=0, high=3)

        start_num = int(np.random.uniform(0, 1) * len(data[0]))
        if start_num > (len(data[0]) - batch_size):
            start_num -= batch_size

        X_batch = get_next(sin_num, data, start_num, batch_size, window_size)
        X_batch = shuffle_x(X_batch)
        Z_batch = sample_Z(batch_size, 1)


        # vars_gmodel_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        # vars_dmodel_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")
        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

        # vars_gmodel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        # vars_dmodel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")


        gen_loss_arr.append(gloss)
        disc_loss_arr.append(dloss)

        # time.sleep(0.02)
        # out.update(progress(i, iterations))
        #     print(sess.run(G_sample, feed_dict={Z: Z_batch}))
        if i % 10 == 0:
            print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, np.mean(dloss), np.mean(gloss)))
            # print("Iterations: %d\t Generator loss: %.4f." % (i, gloss))
            plt.close()
            clear_output(wait=True)

            test_batch = sample_Z(batch_size, 1)
            w = sess.run(G_sample, feed_dict={Z: test_batch})
            # for xzx in range(0, len(w)):
            #     plt.plot(w[xzx])
            plt.plot(w[0])
            plt.plot(X_batch[0])
            plt.plot(pd.DataFrame(w[0][0:64]).rolling(5).mean()[5:])
            # plt.plot(w[:,0])
            w_0 = np.reshape(w[0], newshape=(1, 64))
            print(sess.run(d_real, feed_dict={X: w_0}))
            # print(sess.run(d_fake, feed_dict={Z: w_feed}))
            # for zxz in range(0, len(X_batch)):
            #     plt.plot(X_batch[zxz])

            # plt.plot(answer[0][0])
            # plt.plot(pd.DataFrame(answer[0][0][0:64]).rolling(5).mean()[5:])
            plt.show(block=False)
            time.sleep(1)
            plt.close('all')

    # #tomorrow check out making larger window size
    #     #also check effect of network size
    #     #also check effect of normalisation

    print("--- %s seconds ---" % (time.time() - start_time_gpu))

    # test_batch = np.random.uniform(-1., 1., size=[10, 1, 1])
    # answer = sess.run(G_sample, feed_dict={Z: test_batch})
    # plt.figure()
    # plt.plot(answer[0])
    # print(answer.shape)
    # plt.plot(np.reshape(answer, newshape=(2560, 1))[0:512])
    # plt.plot(pd.DataFrame(np.reshape(answer, newshape=(2560, 1))[0:512]).rolling(5).mean()[5:])
    # plt.show()

    answer = []
    for i in range(0, 4):
        test_batch = np.random.uniform(-1., 1., size=[10, 1])
        answer.append(sess.run(G_sample, feed_dict={Z: test_batch}))
    plt.plot(answer[0])
    plt.plot(answer[1])
    plt.plot(answer[2])
    plt.plot(answer[3])
    plt.show()

    plt.figure()
    plt.title('G and D Losses')
    plt.plot(disc_loss_arr,color='r',label='D')
    plt.plot(gen_loss_arr,color='b',label='G')
    plt.legend()
    plt.show()
    plt.close()

    # print(answer)
    print(
        "*********************************************************************************************************************")

    sess.close()

if __name__ == '__main__':
    main()