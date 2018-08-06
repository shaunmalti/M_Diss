import os
import sys

import pandas as pd
import pandas_datareader.data as web
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler

from numpy import cumsum, log, polyfit, sqrt, std, subtract
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
import csv


p = print



def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return

def hurst(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m = polyfit(log(lags), log(tau), 1)
    return m[0] * 2.0

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper, MultiRNNCell
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def def_placeholders(batch_size, seq_length, latent_dim, num_features, cond=False):
    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_features])
    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])

    # 3 BECAUSE WILL FEED FREQUENCY OF DESIRED AND AMPLITUDE HIGH/LOW
    if cond is True:
        CG = tf.placeholder(tf.float32, [batch_size, 1])
        CD = tf.placeholder(tf.float32, [batch_size, 1])
        return Z, X, CG, CD
    return Z, X


def def_loss(Z, X, seq_length, batch_size, cond_option=False, CG=None, CD=None):
    if cond_option is False:
        G_sample = generator(Z, seq_length, batch_size)
        D_real, D_logit_real = discriminator(X, seq_length, batch_size)
        D_fake, D_logit_fake = discriminator(G_sample, seq_length, batch_size, reuse=True)
    else:
        G_sample = generator(Z, seq_length, batch_size, CG, cond_option)
        D_real, D_logit_real = discriminator(X, seq_length, batch_size, CD, cond_option)
        D_fake, D_logit_fake = discriminator(G_sample, seq_length, batch_size, CG, cond_option, reuse=True)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)
    D_loss = D_loss_real + D_loss_fake

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)

    return D_loss, G_loss


def generator(Z, seq_length, batch_size, CG=None, cond=False,  num_generated_features=1 ,hidden_units_g=100, reuse=False, learn_scale=True):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        W_out_G_initializer = tf.truncated_normal_initializer()
        b_out_G_initializer = tf.truncated_normal_initializer()
        lengths = [1000,10000]
        lstm_initializer = None

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features],
                                  initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        if cond is True:
            condition = tf.stack([CG] * seq_length, axis=1)
            inputs = tf.concat([Z, condition], axis=2)
        else:
            inputs = Z
        # TODO testing
        seq_length_new = tf.placeholder(tf.int32, [None])
        """TODO here LSTMCell was changed to a MultiRNNCell with stacked LSTM layers,
        this seems to 'converge' faster than the original solution - without conditioning
        with conditioning also works but takes slightly longer to converge"""
        cell = LSTMCell(num_units=hidden_units_g, state_is_tuple=True, initializer=lstm_initializer, reuse=reuse)
        # cell = MultiRNNCell([create_cell(2, reuse, lstm_initializer), create_cell(10, reuse, lstm_initializer),
        #                      create_cell(100, reuse, lstm_initializer)], state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                                     sequence_length=[seq_length]*batch_size, inputs=inputs)

        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d

def create_cell(size, reuse, lstm_init):
    lstm_cell = LSTMCell(size,forget_bias=0.7, state_is_tuple=True, initializer=lstm_init, reuse=reuse)
    lstm_cell = DropoutWrapper(lstm_cell)
    return lstm_cell


def discriminator(X, seq_length, batch_size, CD=None, cond=False, hidden_units_d=100, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                                  initializer=tf.truncated_normal_initializer())
        b_out_D = tf.get_variable(name='b_out_D', shape=1,
                                  initializer=tf.truncated_normal_initializer())
        if cond is True:
            condition = tf.stack([CD] * seq_length, axis=1)
            inputs = tf.concat([X, condition], axis=2)
        else:
            inputs = X

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=reuse)
        # (28, 30, 100)

        # cell = MultiRNNCell(
        #     [LSTMCell(30,forget_bias=0.7, state_is_tuple=True, reuse=reuse)], state_is_tuple=True)
        # (28, 30, 1)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)
        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
        # output = tf.nn.sigmoid(logits)

        # logits = tf.matmul(tf.squeeze(rnn_outputs),W_out_D) + b_out_D

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

def get_next_batch(batch_num, batch_size, samples, cond=False, cond_array=None):
    sample_num_start = batch_num * batch_size
    sample_num_end = sample_num_start + batch_size
    if cond is True:
        return samples[sample_num_start:sample_num_end], cond_array[sample_num_start:sample_num_end]
    else:
        return samples[sample_num_start:sample_num_end]


def train_epochs(sess, samples, batch_size, seq_length, latent_dim, D_solver, G_solver,
                 X, Z, D_loss, G_loss, G_sample, epoch, scalar, option, CG=None, CD=None, cond_array=None, cond_option=False):
    # TODO this was changed - 5-1 to 1-5 - see effect
    D_rounds = 5
    G_rounds = 1

    # for i in range(0,int(len(samples) / batch_size)):
    for i in range(0, 450):
    # for i in range(0,10):
        # update discriminator
        for d in range(D_rounds):
            if cond_option is True:
                X_batch, Y_batch = get_next_batch(i, batch_size, samples, cond_option, cond_array)
            else:
                X_batch = get_next_batch(i, batch_size, samples)
            Z_batch = sample_Z(batch_size, seq_length, latent_dim)
            if cond_option is True:
                # TODO this should be cond dimensionality (second 1)
                Y_batch = Y_batch.reshape(-1, 1)
                _ = sess.run(D_solver, feed_dict={X: X_batch, Z: Z_batch, CD: Y_batch, CG: Y_batch})
            else:
                _ = sess.run(D_solver, feed_dict={X: X_batch, Z: Z_batch})

        # update generator
        for g in range(G_rounds):
            if cond_option is True:
                X_batch, Y_batch = get_next_batch(i, batch_size, samples, cond_option, cond_array)
                # TODO this should be cond dimensionality (second 1)
                Y_batch = Y_batch.reshape(-1, 1)
                _ = sess.run(G_solver, feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim), CG: Y_batch})
            else:
                _ = sess.run(G_solver,feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})

        # get loss
        if cond_option is True:
            D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_batch,
                                                                             Z: sample_Z(batch_size, seq_length, latent_dim),
                                                                             CG: Y_batch, CD: Y_batch})
            D_loss_curr = np.mean(D_loss_curr)
            G_loss_curr = np.mean(G_loss_curr)
        else:
            D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_batch,
                                                                             Z: sample_Z(batch_size, seq_length, latent_dim)})
            D_loss_curr = np.mean(D_loss_curr)
            G_loss_curr = np.mean(G_loss_curr)

        if i % 50 == 0 and i != 0:
            print(
                "Iteration: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (i, D_loss_curr, G_loss_curr))
        # if i % 100 == 0 and i != 0:
        if i % 100 == 0:
            if cond_option == True:
                answer = sess.run(G_sample, feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim),
                                                   CG: (np.random.choice([0.5], size=(batch_size, 1)))})
            else:
                # TODO batch_size changed to 1
                answer = sess.run(G_sample, feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})
            answer = scalar.inverse_transform(np.reshape(answer,newshape=(batch_size,seq_length)))
            """Autoregressive = 3
                Moving Average = 2
                AutoReg Mov Avg = 1"""
            if option == 1:
                np.savetxt("Move_Avg_AutoReg.csv", answer, delimiter=',')
            elif option == 2:
                np.savetxt("Move_Avg.csv", answer, delimiter=',')
            elif option == 3:
                np.savetxt("Regressive_2.csv", answer, delimiter=',')

    return D_loss_curr, G_loss_curr


def run(data, scalar, cond_option, option):
    """
    OPTION KEY
    Autoregressive = 3
    Moving Average = 2
    AutoReg Mov Avg = 1"""
    batch_size = 10
    seq_length = 100
    latent_dim = 1
    num_features = 1
    learning_rate = 0.1
    num_epochs = 100

    if cond_option is True:
        Z, X, CG, CD = def_placeholders(batch_size, seq_length, latent_dim, num_features, cond_option)
        D_loss, G_loss = def_loss(Z, X, seq_length, batch_size, cond_option, CG, CD)
        G_sample = generator(Z, seq_length, batch_size, CG, cond_option, reuse=True)
    else:
        Z, X = def_placeholders(batch_size, seq_length, latent_dim, num_features)
        D_loss, G_loss = def_loss(Z, X, seq_length, batch_size)
        G_sample = generator(Z, seq_length, batch_size, reuse=True)

    D_solver, G_solver = def_opt(D_loss, G_loss, learning_rate)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    D_loss_arr = []
    G_loss_arr = []

    for epoch in range(num_epochs):
        if cond_option is True:
            time_before = time.time()
            D_loss_curr, G_loss_curr = train_epochs(sess, data, batch_size, seq_length, latent_dim,
                                                    D_solver, G_solver, X, Z, D_loss, G_loss, G_sample,
                                                    epoch, scalar, option, CG, CD, cond_array, cond_option)
        else:
            time_before = time.time()
            D_loss_curr, G_loss_curr = train_epochs(sess, data, batch_size, seq_length, latent_dim,
                                                    D_solver, G_solver, X, Z, D_loss, G_loss, G_sample,
                                                    epoch, scalar, option)
        print("Epoch Time: ", time.time()-time_before)
        D_loss_arr.append(D_loss_curr)
        G_loss_arr.append(G_loss_curr)
        print("Epoch: %d\t Discriminator loss: %.4f\t Generator loss: %.4f." % (epoch, D_loss_curr, G_loss_curr))
        # shuffle the training data
        perm = np.random.permutation(data.shape[0])
        data = data[perm]
        if cond_option == True:
            cond_array = cond_array[perm]
        if epoch != 0:
            fig = plt.figure()
            plt.title('G and D Losses')
            plt.plot(D_loss_arr, color='r', label='D')
            plt.plot(G_loss_arr, color='b', label='G')
            plt.legend()
            name = "./Plots/epoch_" + str(epoch) + "_losses_move_avg_autoreg.png"
            fig.savefig(name, dpi=fig.dpi)

    save_path = saver.save(sess, "./Models/model_test.ckpt")
    print("Model saved in path: %s" % save_path)



def get_data(option, conditional=False):
    if option == 1:
        alphas = np.array([0.5, -0.25])
        betas = np.array([0.5, -0.3])
        ar = np.r_[1, -alphas]
        ma = np.r_[1, betas]
        n = int(100)
        burn = int(n / 10)  # number of samples to discard before fit
        arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)

        samples = []
        cond_arr = []
        n_lists = 10000
        for j in range(n_lists):
            signals = []
            cond_sig = []
            signals.append(smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn))
            for i in range(0, n):
                cond_sig.append([alphas,betas])
            samples.append(np.array(signals).T)
            cond_arr.append(np.array(cond_sig).T)
        samples = np.array(samples)
        scalar = MinMaxScaler(feature_range=(-1, 1))
        samples = np.reshape(samples, newshape=(n_lists, n))
        scalar.fit(samples)

        samples = scalar.transform(samples)
        samples = np.reshape(samples, newshape=[len(samples), n, 1])
        if conditional == True:
            return samples, cond_arr, scalar
        else:
            return samples, scalar

    elif option == 2:

        n_samples = int(100)
        n = int(100)
        b = 0.6
        alphas = np.array([0.])
        betas = np.array([0.6])

        # add zero-lag and negate alphas
        ar = np.r_[1, -alphas]
        ma = np.r_[1, betas]

        samples = []
        cond_arr = []
        n_lists = 10000
        for j in range(n_lists):
            # x = w = np.random.normal(size=n_samples)
            signals = []
            cond_sig = []
            signals.append(smt.arma_generate_sample(ar=ar, ma=ma, nsample=n))
            if conditional == True:
                for i in range(0, n):
                    cond_sig.append(b)
            samples.append(np.array(signals).T)
            cond_arr.append(np.array(cond_sig).T)
        samples = np.array(samples)
        cond_arr = np.array(cond_arr)
        scalar = MinMaxScaler(feature_range=(-1, 1))
        samples = np.reshape(samples, newshape=(n_lists, n_samples))
        scalar.fit(samples)

        samples = scalar.transform(samples)
        samples = np.reshape(samples, newshape=[len(samples), n_samples, 1])
        if conditional == True:
            return samples, cond_arr, scalar
        else:
            return samples, scalar

    elif option == 3:
        # autoregressive model
        # dependent variable is regressed against one or more lagged values
        samples = []
        cond_arr = []
        n_lists = 10000
        n_samples = int(100)
        a = 0.6
        for j in range(n_lists):
            x = w = np.random.normal(size=n_samples)
            signals = []
            cond_sig = []
            for j in range(n_samples):
                signals.append(a * x[j - 1] + w[j])
                if conditional == True:
                    cond_sig.append(a)
            samples.append(np.array(signals).T)
            cond_arr.append(np.array(cond_sig).T)
        samples = np.array(samples)
        cond_arr = np.array(cond_arr)
        scalar = MinMaxScaler(feature_range=(-1, 1))
        scalar.fit(samples)
        samples = scalar.transform(samples)

        samples = np.reshape(samples, newshape=[len(samples), n_samples, 1])
        if conditional == True:
            return samples, cond_arr, scalar
        else:
            return samples, scalar


def main():
    """Autoregressive = 3
        Moving Average = 2
        AutoReg Mov Avg = 1"""
    option = 2
    conditional = False
    if conditional == True:
        samples, cond_arr, scalar = get_data(option, conditional)
    else:
        samples, scalar = get_data(option)
    run(samples, scalar, conditional, option)




if __name__ == '__main__':
    main()