import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def def_placeholders(batch_size, seq_length, latent_dim, num_features):
    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_features])
    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    return Z, X


def def_loss(Z, X, seq_length, batch_size):
    G_sample = generator(Z, seq_length, batch_size)
    D_real, D_logit_real = discriminator(X, seq_length, batch_size)
    D_fake, D_logit_fake = discriminator(G_sample, seq_length, batch_size, reuse=True)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)
    D_loss = D_loss_real + D_loss_fake

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)

    return D_loss, G_loss


def generator(Z, seq_length, batch_size, num_generated_features=1 ,hidden_units_g=100, reuse=False, learn_scale=True):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        W_out_G_initializer = tf.truncated_normal_initializer()
        b_out_G_initializer = tf.truncated_normal_initializer()
        scale_out_G_initializer = tf.constant_initializer(value=1.0)
        lstm_initializer = None
        bias_start = 1.0

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features],
                                  initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer,
                                      trainable=learn_scale)

        inputs = Z

        cell = LSTMCell(num_units=hidden_units_g, state_is_tuple=True, initializer=lstm_initializer, reuse=reuse)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                                    sequence_length=[seq_length]*batch_size, inputs=inputs)

        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d


def discriminator(X, seq_length, batch_size, hidden_units_d=100, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                initializer=tf.truncated_normal_initializer())
        b_out_D = tf.get_variable(name='b_out_D', shape=1,
                initializer=tf.truncated_normal_initializer())
        inputs = X

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)

        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
        output = tf.nn.sigmoid(logits)

    return output, logits


def def_opt(D_loss, G_loss, learning_rate):
    disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

    D_loss_mean_over_batch = tf.reduce_mean(D_loss)
    D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=disc_vars)
    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss_mean_over_batch, var_list=gen_vars)
    return D_solver, G_solver


def sample_Z(batch_size, seq_length, latent_dim):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    return sample

def get_next_batch(batch_num, batch_size, samples):
    sample_num_start = batch_num * batch_size
    sample_num_end = sample_num_start + batch_size
    return samples[sample_num_start:sample_num_end]


def train_epochs(sess, samples, batch_size, seq_length, latent_dim, D_solver, G_solver,
                 X, Z, D_loss, G_loss, G_sample, epoch):
    D_rounds = 5
    G_rounds = 1

    for i in range(0,int(len(samples) / batch_size)):
    # for i in range(0, 10):
        # update discriminator
        for d in range(D_rounds):
            X_batch = get_next_batch(i, batch_size, samples)
            Z_batch = sample_Z(batch_size, seq_length, latent_dim)

            _ = sess.run(D_solver, feed_dict={X: X_batch, Z: Z_batch})

        # update generator
        for g in range(G_rounds):
            _ = sess.run(G_solver,feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})

        D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_batch,
                                                                         Z: sample_Z(batch_size, seq_length, latent_dim)})
        D_loss_curr = np.mean(D_loss_curr)
        G_loss_curr = np.mean(G_loss_curr)

        if i % 50 == 0 and i != 0:
            print("Iteration: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, D_loss_curr, G_loss_curr))
        if i % 100 == 0 and i != 0:
            answer = sess.run(G_sample, feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})
            fig = plt.figure()
            plt.plot(answer[0])
            name = "./Plots/epoch_" + str(epoch) + "_iter_" + str(i) + "_test.png"
            fig.savefig(name, dpi=fig.dpi)
            plt.show(block=False)
            time.sleep(3)
            plt.close('all')

    return D_loss_curr, G_loss_curr


def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1,
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    samples = np.array(samples)
    return samples


def main():
    batch_size = 28
    seq_length = 30
    latent_dim = 3
    num_features = 1
    learning_rate = 0.1
    num_epochs = 100
    samples = sine_wave()

    Z, X = def_placeholders(batch_size, seq_length, latent_dim, num_features)

    D_loss, G_loss = def_loss(Z, X, seq_length, batch_size)

    D_solver, G_solver = def_opt(D_loss, G_loss, learning_rate)

    G_sample = generator(Z, seq_length, batch_size, reuse=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Z_batch = sample_Z(batch_size, seq_length, latent_dim)

    # Vis_sample = sess.run(G_sample, feed_dict={Z: Z_batch})

    D_loss_arr = []
    G_loss_arr = []

    for epoch in range(num_epochs):
        D_loss_curr, G_loss_curr = train_epochs(sess, samples, batch_size, seq_length, latent_dim,
                                                D_solver, G_solver, X, Z, D_loss, G_loss, G_sample, epoch)

        D_loss_arr.append(D_loss_curr)
        G_loss_arr.append(G_loss_curr)
        print("Epoch: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (epoch, D_loss_curr, G_loss_curr))
        # shuffle the training data
        perm = np.random.permutation(samples.shape[0])
        samples = samples[perm]

        fig = plt.figure()
        plt.title('G and D Losses')
        plt.plot(D_loss_arr, color='r', label='D')
        plt.plot(G_loss_arr, color='b', label='G')
        plt.legend()
        name = "./Plots/epoch_" + str(epoch) + "_losses.png"
        fig.savefig(name, dpi=fig.dpi)
        plt.show(block=False)
        time.sleep(3)
        plt.close('all')

if __name__ == '__main__':
    main()