import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib import rnn

def import_csv():
    # import, set header names and change columns depending on content
    data = pd.read_csv('CD.txt',sep=',',names=['Date','Open','High','Low','Close','Volume'])

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

    scaler = MinMaxScaler(feature_range=(-1,1))

    array = scaler.fit_transform(np.reshape(np.array(new_array),newshape=(len(new_array),1)))

    return array, scaler

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(Z,reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        rnn_cell = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(1,activation=None,state_is_tuple=True),
             rnn.BasicLSTMCell(5,activation=None,state_is_tuple=True),
             rnn.BasicLSTMCell(2, activation=None, state_is_tuple=True)])
        outputs, states = rnn.static_rnn(rnn_cell, [Z], dtype=tf.float32)
        # out = tf.nn.tanh(outputs[-1])
    return outputs[-1]

def discriminator(X,reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        rnn_cell = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(2,activation=None),
             rnn.BasicLSTMCell(5, activation=None),
             rnn.BasicLSTMCell(1,activation=None)])
        outputs, states = rnn.static_rnn(rnn_cell, [X], dtype=tf.float32)
        out = tf.nn.sigmoid(outputs[-1])
    return out

def get_next(all_data,start_num,batch_size):
    return all_data[start_num:(start_num+batch_size)]

def main():

    """Note, making Disc layers stateful leads to negative line being output"""

    # data, scalar = import_csv()
    tf.reset_default_graph()
    data_x = np.arange(0,4000,1,dtype=float).reshape(4000,1)
    data_y = np.arange(0,-4000,-1,dtype=float).reshape(4000,1)
    scalar_x = MinMaxScaler(feature_range=(-1,1))
    scalar_x.fit(data_x)
    scalar_y = MinMaxScaler(feature_range=(-1, 1))
    scalar_y.fit(data_y)

    # data_x = scalar.fit_transform(data_y.reshape(-1,1))
    # data_y = scalar.fit_transform(data_y.reshape(-1,1))

    data = np.concatenate([data_x, data_y], axis=1)

    X = tf.placeholder(tf.float32, [None, 2])
    Z = tf.placeholder(tf.float32, [None, 2])

    G_sample = generator(Z)
    r_logits = discriminator(X)
    f_logits = discriminator(G_sample, reuse=True)

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)) +
                               tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
    disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step

    # sess = tf.Session(config=config)
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    batch_size = 256
    nd_steps = 10
    ng_steps = 10

    gen_loss_arr = []
    disc_loss_arr = []

    for i in range(10000):
        start_num = int(np.random.uniform(0, 1) * len(data))
        if start_num > (len(data) - batch_size):
            start_num -= batch_size
        # if i == 0:

        X_batch = get_next(data,start_num,batch_size)
        Z_batch = sample_Z(batch_size, 2)

        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})


        # print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
        gen_loss_arr.append(gloss)
        disc_loss_arr.append(dloss)
        if i % 100 == 0:
            print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, dloss, gloss))
            answer = sess.run(G_sample, feed_dict={Z: Z_batch})
            ans_y = answer[:, 1]
            ans_x = answer[:, 0]
            ans_y_re = scalar_x.inverse_transform(ans_y.reshape(-1, 1))
            ans_x_re = scalar_y.inverse_transform(ans_x.reshape(-1, 1))
            # plt.scatter(ans_x,ans_y)
            plt.scatter(ans_x_re,ans_y_re)
            # plt.plot(np.reshape(answer,newshape=(batch_size,1)),color='r',label='Pred')
            # plt.plot(np.reshape(x_batch,newshape=(batch_size,1)),color='b',label='Actual')
            plt.title('Prediction Iter: {0}'.format(i))
            plt.savefig('./iteration_pred_{0}'.format(i))
            plt.close()

    plt.plot(disc_loss_arr, color='r', label='D')
    plt.plot(gen_loss_arr, color='b', label='G')
    plt.legend()
    plt.show()

    answer = []

    test_batch = sample_Z(batch_size, 2)
    # answer = sess.run(G_sample, feed_dict={Z: test_batch})
    # print(sess.run(f_logits,feed_dict={X: answer}))
    # print(sess.run(g_rep, feed_dict={Z: answer}))
    # plt.scatter(np.arange(0,len(answer),1),scalar.inverse_transform(answer).reshape(-1,1))
    answer = sess.run(G_sample, feed_dict={Z: Z_batch})
    ans_y = answer[:, 1]
    ans_x = answer[:, 0]
    # ans_y_re = scalar.inverse_transform(ans_y.reshape(-1, 1))
    # ans_x_re = scalar.inverse_transform(ans_x.reshape(-1, 1))
    plt.scatter(ans_x,ans_y)
    # plt.plot(scalar.inverse_transform(answer).reshape(-1,1))
    plt.show()

if __name__ == '__main__':
    main()

    """THEREFORE ADDING THE SECOND DIMENSION - X AXIS - LEADS TO A GOOD RESULT"""