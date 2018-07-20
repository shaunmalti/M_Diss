import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)


def generator(Z, reuse=False):
    # with tf.variable_scope("GAN/Generator",reuse=reuse):
    #     h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
    #     h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
    #     out = tf.layers.dense(h2,1)


    nodes_1 = 1
    hidden_nodes = 256
    Z = tf.unstack(Z, 1, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(nodes_1,activation=tf.nn.relu,state_is_tuple=True),
                                 rnn.BasicLSTMCell(hidden_nodes,activation=tf.nn.relu,state_is_tuple=True),
                                 rnn.BasicLSTMCell(1,activation=tf.nn.relu)])
    with tf.variable_scope("GAN/Generator",reuse=reuse) as gen:
        weights = tf.Variable(tf.random_normal([nodes_1,1]))
        biases = tf.Variable(tf.random_normal([1]))
        outputs, states = rnn.static_rnn(rnn_cell, Z, dtype=tf.float32)
        res = tf.matmul(outputs[-1], weights) + biases
        out = tf.layers.dense(res,1)

        g_params = [v for v in tf.global_variables() if v.name.startswith(gen.name)]

    # TODO what does this do
    with tf.name_scope('gen_params'):
        for param in g_params:
            variable_summaries(param)
    return out

def discriminator(X, reuse=False):
    # with tf.variable_scope("GAN/Discriminator",reuse=reuse):
    #     h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
    #     h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
    #     h3 = tf.layers.dense(h2,2)
    #     out = tf.layers.dense(h3,1)
    #
    # return out, h3
    nodes_1 = 1
    hidden_nodes = 2
    batch_size = 256
    X = tf.unstack(X, 1, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(nodes_1,activation=None)])
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        weights = tf.Variable(tf.random_normal([nodes_1, 1]))
        biases = tf.Variable(tf.random_normal([1]))
        outputs, states = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)
        res = tf.matmul(outputs[-1], weights) + biases
        y_data = tf.nn.sigmoid(tf.slice(res, [0, 0],[-1, -1],name=None))
        y_generated =  tf.nn.sigmoid(tf.slice(res, [batch_size, 0], [-1, -1], name=None))
    return y_data, y_generated

def sample_data():
    # DEFINE SEQUENCES OF DATA
    seq_2 = [10, 1, 10, 1]
    seq_1 = [10, 1, 10, 1]

    # tf.reset_default_graph()
    # np.random.seed(1)
    data = np.random.choice([1, 2], size=1000)
    # IMPORTANT - given a test of just repetitions of -1,1,1,1. Would the GAN create long runs of positive
    # numbers with just one negative
    # data = np.random.choice([1],size=100)

    unique, counts = np.unique(data, return_counts=True)
    data_dict = dict(zip(unique, counts))
    all_data = []
    for i in range(len(data)):
        if data[i] == 1:
            all_data.append(seq_1)
        else:
            all_data.append(seq_2)

    all_data = np.ndarray.flatten(np.array(all_data))

    return all_data

def get_next(all_data,start_num,batch_size):
    return all_data[start_num:(start_num+batch_size)]

def sample_Z(m):
    return np.random.uniform(-1., 1., size=[m,1])

def main():
    data = sample_data()

    X = tf.placeholder(tf.float32, [None,1, 1], name='real')
    # Z = tf.placeholder(tf.float32, [None, 1])
    Z = tf.placeholder(tf.float32, [None,1, 1], name='noise')
    # test = tf.placeholder(tf.float32,[None,1,1])

    G_sample = generator(Z)
    G_sample = tf.reshape(G_sample, shape=(-1, 1, 1))

    r_logits, r_rep = discriminator(X)
    f_logits, g_rep = discriminator(G_sample, reuse=True)

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
        r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

    # gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
    # disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step

    gen_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)
    disc_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)

    # TEST
    test = tf.nn.leaky_relu(G_sample)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    batch_size = 256
    nd_steps = 10
    ng_steps = 10
    # for i in range(10001):

    gen_loss_arr = []
    disc_loss_arr = []

    for i in range(1000):
        start_num = int(np.random.uniform(0,1)*len(data))
        if start_num > (len(data)-batch_size):
            start_num -= batch_size

        X_batch = np.reshape(get_next(data,start_num,batch_size),newshape=(256,1,1))
        Z_batch = np.reshape(sample_Z(batch_size),newshape=(256,1,1))

        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

        rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        test_batch = np.reshape(sample_Z(batch_size), newshape=(256, 1, 1))
        answer = sess.run(G_sample, feed_dict={Z: test_batch})
        answer_test = sess.run(test,feed_dict={Z: test_batch})
        # ans_d = sess.run(r_logits,feed_dict={X: answer})
        ans_d = sess.run(f_logits, feed_dict={Z: answer})
        print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f, Example Prediction: %.4f, Corresponding Discr Pred: %.4f, other: %.4f." % (i, dloss,gloss,answer[-1],ans_d[-1],answer_test[-1]))
        gen_loss_arr.append(gloss)
        disc_loss_arr.append(dloss)

        # if i % 100 == 0:
        #     g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})

    test_batch = np.reshape(sample_Z(batch_size),newshape=(256,1,1))
    answer = sess.run(G_sample,feed_dict={Z:test_batch})
    ans_d = sess.run(f_logits, feed_dict={Z: answer})
    plt.title('G and D Losses')
    plt.plot(disc_loss_arr,color='r',label='D')
    plt.plot(gen_loss_arr,color='b',label='G')
    plt.legend()
    plt.show()
    f = open('ans.csv','w')
    f.write(str(answer))

    plt.title('G and D Outputs')
    plt.plot(np.reshape(ans_d,newshape=(256)),color='r',label='D')
    plt.plot(np.reshape(answer,newshape=(256)),color='b',label='G')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()