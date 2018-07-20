import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import tensorflow as tf
from tensorflow.contrib import rnn

def import_csv():
    # import, set header names and change columns depending on content
    data = pd.read_csv('C.txt',sep=',',names=['Date','Open','High','Low','Close','Volume'])

    # redefine data as being 0/1. 1 when close > open and 0 in all other cases
    data_np = data.values
    new_array = []

    for i in range(0,len(data_np)):
        if data_np[i][4] > data_np[i][1]:
            new_array.append(1)
        else:
            new_array.append(0)

    return new_array

def G(z, batch_size, hidden_units_g=1,reuse=False):
    # TODO change init of weights to zeros, change number of layers, change init to tf.Var
    with tf.variable_scope("generator") as scope:
        z = tf.reshape(z, shape=(batch_size, 1))
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_units_g),rnn.BasicLSTMCell(hidden_units_g)])
        outputs,states = rnn.static_rnn(rnn_cell,[z],dtype=tf.float32)
        # output = tf.nn.tanh(outputs[-1])
    return outputs

def D(x,batch_size,hidden_units_d=1,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            # scope.reuse_variables()
            tf.get_variable_scope().reuse_variables()
        x = tf.reshape(x,shape=(batch_size,1))

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_units_d),rnn.BasicLSTMCell(hidden_units_d)])
        outputs,states = rnn.static_rnn(rnn_cell,[x],dtype=tf.float32)
        output = tf.nn.sigmoid(outputs)
        output = tf.reshape(output, (batch_size, 1))
    return output


def hurst(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)

    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    m = polyfit(log(lags), log(tau), 1)

    return m[0] * 2.0

def get_next(all_data,start_num,batch_size):
    return all_data[start_num:(start_num+batch_size)]

def main():
    tf.reset_default_graph()
    sess = tf.Session()

    # import financial data and perform close open difference
    data = import_csv()

    # calculate herst component of data
    print(hurst(data))

    batch_size = 256

    X = tf.placeholder(tf.float32, [batch_size, 1], name='real')
    Z = tf.placeholder(tf.float32, [batch_size, 1], name='noise')

    G_sample = G(Z, batch_size)

    r_rep = D(X, batch_size)
    g_rep = D(G_sample, batch_size, reuse=True)

    # defining loss measures
    g_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(g_rep), predictions=g_rep))

    d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(r_rep), predictions=r_rep))
    d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(g_rep), predictions=g_rep))
    d_loss = d_loss_fake + d_loss_real

    # getting vars for training
    t_vars = tf.trainable_variables()
    disc_vars = [var for var in t_vars if 'discriminator' in var.name]
    gen_vars = [var for var in t_vars if 'generator' in var.name]

    # defining optimiser
    # with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    #     print("reuse or not: {}".format(tf.get_variable_scope().reuse))
    #     assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"
    trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list=disc_vars)
    trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list=gen_vars)


    # TRAINING PORTION
    batch_size = 256
    sess.run(tf.global_variables_initializer())
    iterations = 3000
    gen_loss_arr = []
    disc_loss_arr = []
    for i in range(iterations):
        start_num = int(np.random.uniform(0, 1) * len(data))
        if start_num > (len(data) - batch_size):
            start_num -= batch_size

        z_batch = np.random.normal(-1, 1, size=[batch_size, 1])
        x_batch =  np.reshape(get_next(data,start_num,batch_size),newshape=(batch_size,1))
        _, dloss = sess.run([trainerD, d_loss], feed_dict={X: x_batch,Z: z_batch})  # Update the discriminator
        _, gloss = sess.run([trainerG, g_loss], feed_dict={Z: z_batch})  # Update the generator
        gen_loss_arr.append(gloss)
        disc_loss_arr.append(dloss)
        if i % 10 == 0:
            print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, dloss, gloss))


    plt.figure()
    plt.title('G and D Losses')
    plt.plot(disc_loss_arr,color='r',label='D')
    plt.plot(gen_loss_arr,color='b',label='G')
    plt.legend()
    plt.show()

    test_batch = np.random.normal(-1, 1, size=[batch_size, 1])
    answer = sess.run(G_sample, feed_dict={Z: test_batch})
    answer = np.reshape(answer,newshape=(256,1))
    print(hurst(answer))

if __name__ == '__main__':
    main()