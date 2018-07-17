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
    return np.random.uniform(-1., 1., size=[m,1])


def G(z, batch_size, hidden_units_g=1,num_generated_features=1,parameters=None,learn_scale=True,reuse=False):
    # TODO change init of weights to zeros, change number of layers, change init to tf.Var
    # with tf.variable_scope("generator") as scope:
    #     seq_length = batch_size
    #     # if reuse:
    #     #     scope.reuse_variables()
    #     if parameters is None:
    #         W_out_G_initializer = tf.truncated_normal_initializer()
    #         b_out_G_initializer = tf.truncated_normal_initializer()
    #         # W_out_G_initializer = tf.zeros_initializer()
    #         # b_out_G_initializer = tf.zeros_initializer()
    #         scale_out_G_initializer = tf.constant_initializer(value=1.0)
    #         lstm_initializer = None
    #         bias_start = 1.0
    #     else:
    #         W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
    #         b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
    #         try:
    #             scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
    #         except KeyError:
    #             scale_out_G_initializer = tf.constant_initializer(value=1)
    #             # assert learn_scale
    #         lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
    #         bias_start = parameters['generator/rnn/lstm_cell/biases:0']
    #
    #     W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features],
    #                               initializer=W_out_G_initializer)
    #     b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
    #     scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer,
    #                                   trainable=learn_scale)
    #     # if cond_dim > 0:
    #     #     # CGAN!
    #     #     assert not c is None
    #     #     repeated_encoding = tf.stack([c] * seq_length, axis=1)
    #     #     inputs = tf.concat([z, repeated_encoding], axis=2)
    #     #
    #     #     # repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
    #     #     # repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1], cond_dim])
    #     #     # inputs = tf.concat([repeated_encoding, z], 2)
    #     # else:
    #     inputs = z
    #
    #     cell = LSTMCell(num_units=hidden_units_g,
    #                     state_is_tuple=True,
    #                     initializer=lstm_initializer,
    #                     reuse=reuse)
    #     rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
    #         cell=cell,
    #         dtype=tf.float32,
    #         # sequence_length= [1] * batch_size,
    #         sequence_length= [seq_length] * batch_size,
    #         inputs=inputs)
    #     rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
    #     logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
    #     # TODO why is the output tanh-ed?
    #     # output_2d = tf.nn.tanh(logits_2d)
    #     output_2d = logits_2d
    #     output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    # return output_3d

    # weights = {
    #     'out': tf.Variable(tf.random_normal([hidden_units_g,1]))
    # }
    # biases = {
    #     'out': tf.Variable(tf.random_normal([hidden_units_g,1]))
    # }

    w_1 = tf.get_variable(name='W_1', shape=[hidden_units_g, 1],initializer=tf.truncated_normal_initializer())
    b_1 = tf.get_variable(name='B_1', shape=[hidden_units_g, 1],initializer=tf.truncated_normal_initializer())
    z = tf.reshape(z,shape=(256,1))
    with tf.variable_scope("generator") as scope:
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_units_g)])
        outputs,states = rnn.static_rnn(rnn_cell,[z],dtype=tf.float32)

    # return tf.matmul(outputs[-1],w_1)+b_1
        return outputs[-1]

def D(x,batch_size,hidden_units_d=1,reuse=False,batch_mean=False):
    # Simple GAN discriminator
    # model = Sequential(name='Disc')
    #
    # model.add(LSTM(10, input_shape=(1, 1), return_sequences=True, activation=None))
    # model.add(Dense(1,activation='sigmoid'))
    #
    # value = Input(shape=[1,1,])
    # prob = model(value)
    # # (?, 1, 1)
    # return Model(value,prob)
    # TODO from here commented out
    # with tf.variable_scope("discriminator") as scope:
    #     if reuse:
    #         scope.reuse_variables()
    #     W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
    #                               initializer=tf.truncated_normal_initializer())
    #     b_out_D = tf.get_variable(name='b_out_D', shape=1,
    #                               initializer=tf.truncated_normal_initializer())
    #
    #     # W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
    #     #                           initializer=tf.zeros_initializer())
    #     # b_out_D = tf.get_variable(name='b_out_D', shape=1,
    #     #                           initializer=tf.zeros_initializer())
    #
    #     # if cond_dim > 0:
    #     #     assert not c is None
    #     #     repeated_encoding = tf.stack([c] * seq_length, axis=1)
    #     #     inputs = tf.concat([x, repeated_encoding], axis=2)
    #     # else:
    #     inputs = x
    #     # add the average of the inputs to the inputs (mode collapse?
    #     if batch_mean:
    #         mean_over_batch = tf.stack([tf.reduce_mean(x, axis=0)] * batch_size, axis=0)
    #         inputs = tf.concat([x, mean_over_batch], axis=2)
    #
    #     cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d,
    #                                    state_is_tuple=True,
    #                                    reuse=reuse)
    #     rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,inputs=inputs)
    #     logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
    #     output = tf.nn.sigmoid(logits)
    # return output, logits
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # weights = {
        #     'out': tf.Variable(tf.random_normal([hidden_units_d,1]))
        # }
        # biases = {
        #     'out': tf.Variable(tf.random_normal([hidden_units_d,1]))
        # }
        w_1 = tf.get_variable(name='W_1', shape=[hidden_units_d, 1], initializer=tf.truncated_normal_initializer())
        b_1 = tf.get_variable(name='B_1', shape=[hidden_units_d, 1], initializer=tf.truncated_normal_initializer())
        x = tf.reshape(x,shape=(256,1))

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_units_d)])
        outputs,states = rnn.static_rnn(rnn_cell,[x],dtype=tf.float32)
        # ans = tf.matmul(outputs[-1], w_1) + b_1
        # output = tf.nn.sigmoid(ans)
    return outputs[-1]


def get_next(all_data,start_num,batch_size):
    return all_data[start_num:(start_num+batch_size)]

def main():
    # data = ingest_normalise()
    x = np.arange(0, 4000, 1)
    data = np.sin(x)

    batch_size = 256

    X = tf.placeholder(tf.float32, [batch_size, 1,1], name='real')
    # Z = tf.placeholder(tf.float32, [None, 1])
    Z = tf.placeholder(tf.float32, [batch_size, 1,1], name='noise')
    # test = tf.placeholder(tf.float32,[None,1,1])

    G_sample = G(Z,batch_size)
    G_sample = tf.reshape(G_sample, shape=(-1, 1, 1))

    r_rep = D(X,batch_size)
    g_rep = D(G_sample,batch_size, reuse=True)

    # disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
    #     r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    # gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_rep, labels=tf.ones_like(g_rep)))

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_rep, labels=tf.ones_like(r_rep)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_rep, labels=tf.zeros_like(g_rep)))
    d_loss = d_loss_real + d_loss_fake

    # g_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(g_rep),predictions=g_rep))
    #
    # d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(r_rep),predictions=r_rep))
    # d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(g_rep),predictions=g_rep))
    # d_loss = d_loss_fake + d_loss_real

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    # gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
    # disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step

    gen_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(g_loss, var_list=gen_vars)
    disc_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss, var_list=disc_vars)

    # gen_step = tf.train.GradientDescentOptimizer(0.001).minimize(g_loss)
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