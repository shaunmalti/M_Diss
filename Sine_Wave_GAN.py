import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn


def ingest_normalise():
    series = pd.read_csv('sine-wave.csv', header=None)

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(series.values)
    series = pd.DataFrame(scaled)

    # window_size = 10
    #
    # series_s = series.copy()
    # for i in range(window_size):
    #     series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
    #
    # series.dropna(axis=0, inplace=True)
    # # also create window and shuffle
    #
    # series = series.iloc[:, :-1]

    return series.values

def sample_Z(m):
    return np.random.uniform(0., 1., size=[m,1])


def G(z, batch_size, hidden_units_g=1,num_generated_features=1,parameters=None,learn_scale=True,reuse=False):
    # TODO change init of weights to zeros, change number of layers, change init to tf.Var
    with tf.variable_scope("generator") as scope:
        # w_1 = tf.get_variable(name='W_1', shape=[hidden_units_g, 1], initializer=tf.truncated_normal_initializer())
        # b_1 = tf.get_variable(name='B_1', shape=[hidden_units_g, 1], initializer=tf.truncated_normal_initializer())
        z = tf.reshape(z, shape=(256, 1))
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_units_g),rnn.BasicLSTMCell(hidden_units_g)])
        outputs,states = rnn.static_rnn(rnn_cell,[z],dtype=tf.float32)

    # return tf.matmul(outputs[-1],w_1)+b_1
        return outputs[-1]

def D(x,batch_size,hidden_units_d=1,reuse=False,batch_mean=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        # w_1 = tf.get_variable(name='W_1', shape=[hidden_units_d, 1], initializer=tf.truncated_normal_initializer())
        # b_1 = tf.get_variable(name='B_1', shape=[hidden_units_d, 1], initializer=tf.truncated_normal_initializer())
        x = tf.reshape(x,shape=(256,1))

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_units_d),rnn.BasicLSTMCell(hidden_units_d)])
        outputs,states = rnn.static_rnn(rnn_cell,[x],dtype=tf.float32)
        # ans = tf.matmul(outputs[-1], w_1) + b_1
        # output = tf.nn.sigmoid(ans)
    return outputs[-1]


def get_next(all_data,start_num,batch_size):
    return all_data[start_num:(start_num+batch_size)]

def main():
    # data = ingest_normalise()
    x = np.arange(0, 4000, 1)
    data = np.sin(x) + 1

    batch_size = 256

    X = tf.placeholder(tf.float32, [batch_size, 1,1], name='real')
    # Z = tf.placeholder(tf.float32, [None, 1])
    Z = tf.placeholder(tf.float32, [batch_size, 1,1], name='noise')
    # test = tf.placeholder(tf.float32,[None,1,1])

    G_sample = G(Z,batch_size)
    G_sample = tf.reshape(G_sample, shape=(-1, 1, 1))

    r_rep = D(X,batch_size)
    g_rep = D(G_sample,batch_size, reuse=True)

    # defining losses
    # disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
    #     r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    # gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_rep, labels=tf.ones_like(g_rep)))
    #
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_rep, labels=tf.ones_like(r_rep)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_rep, labels=tf.zeros_like(g_rep)))
    d_loss = d_loss_real + d_loss_fake

    # g_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(g_rep),predictions=g_rep))
    #
    # d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(r_rep),predictions=r_rep))
    # d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(g_rep),predictions=g_rep))
    # d_loss = d_loss_fake + d_loss_real

    # getting vars for training
    # gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    # disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
    t_vars = tf.trainable_variables()
    disc_vars = [var for var in t_vars if 'discriminator' in var.name]
    gen_vars = [var for var in t_vars if 'generator' in var.name]

    # defining optimiser
    # gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
    # disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step

    gen_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(g_loss, var_list=gen_vars)
    disc_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss, var_list=disc_vars)

    # gen_step = tf.train.GradientDescentOptimizer(0.01).minimize(g_loss)
    # disc_step = tf.train.GradientDescentOptimizer(0.001).minimize(d_loss)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    # TRAINING PORTION
    nd_steps = 10
    ng_steps = 10

    gen_loss_arr = []
    disc_loss_arr = []


    for i in range(10000):
        start_num = int(np.random.uniform(0,1)*len(data))
        if start_num > (len(data)-batch_size):
            start_num -= batch_size
        # start_num = i

        X_batch = np.reshape(get_next(data,start_num,batch_size),newshape=(batch_size,1,1))
        Z_batch = np.reshape(sample_Z(batch_size),newshape=(batch_size,1,1))

        # tf.summary.histogram("Z_batch", Z_batch)
        # tf.summary.histogram("X_batch", X_batch)

        # merge = tf.summary.merge_all()
        # merge_2 = tf.summary.merge_all()

        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, d_loss], feed_dict={X: X_batch, Z: Z_batch})
        rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, g_loss], feed_dict={Z: Z_batch})
        rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        # test_batch = np.reshape(sample_Z(batch_size), newshape=(10, 1, 1))
        # answer = sess.run(G_sample, feed_dict={Z: test_batch})
        # ans_d = sess.run(f_logits, feed_dict={Z: answer})
        if i % 10 == 0:
            print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, dloss,gloss))
        gen_loss_arr.append(gloss)
        disc_loss_arr.append(dloss)

        if i % 100 == 0 and i != 0:
            plt.title('G and D Losses Iter: {0}'.format(i))
            plt.plot(disc_loss_arr, color='r', label='D')
            plt.plot(gen_loss_arr, color='b', label='G')
            plt.legend()
            plt.savefig('./Figures_SIN_GAN/iteration_loss_{0}'.format(i))
            plt.close()

            test_batch_int = np.reshape(sample_Z(batch_size), newshape=(batch_size, 1, 1))
            answer_int = sess.run(G_sample, feed_dict={Z: test_batch_int})
            plt.title('Prediction Iter: {0}'.format(i))
            plt.plot(np.reshape(answer_int, newshape=(batch_size, 1)))
            plt.savefig('./Figures_SIN_GAN/iteration_pred_{0}'.format(i))
            plt.close()

    plt.figure()
    plt.title('G and D Losses')
    plt.plot(disc_loss_arr,color='r',label='D')
    plt.plot(gen_loss_arr,color='b',label='G')
    plt.legend()
    plt.show()

    # answer = []
    # for i in range(0,10):
        # test_batch = np.reshape(sample_Z(batch_size), newshape=(256 1, 1))
        # answer.append(np.ndarray.flatten(np.array(sess.run(G_sample, feed_dict={Z: test_batch}))))
    # plt.plot(np.ndarray.flatten(np.array(answer)))
    # plt.show()

    test_batch = np.reshape(sample_Z(batch_size), newshape=(batch_size, 1, 1))
    answer = sess.run(G_sample, feed_dict={Z: test_batch})
    plt.figure()
    plt.plot(np.reshape(answer,newshape=(batch_size,1)))
    plt.show()
    print(answer)
    print("*********************************************************************************************************************")
    print(sess.run(r_rep, feed_dict={X: answer}))

if __name__ == '__main__':
    main()