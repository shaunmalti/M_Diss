import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np

def sample_Z(batch_size, seq_length, latent_dim):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    return sample

def rebuild_model(graph, Z):

    g_kernel = graph.get_tensor_by_name('generator/rnn/lstm_g/kernel:0')
    g_bias = graph.get_tensor_by_name('generator/rnn/lstm_g/bias:0')
    w_out_g = graph.get_tensor_by_name('generator/W_out_G:0')
    b_out_g = graph.get_tensor_by_name('generator/b_out_G:0')

    cell = LSTMCell(num_units=100, state_is_tuple=True, initializer=tf.constant_initializer(value=sess.run(g_kernel)))
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=Z)
    rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, 100], name='reshape_output')
    logits_2d = tf.matmul(rnn_outputs_2d, w_out_g) + b_out_g
    output_2d = tf.nn.tanh(logits_2d, name='tanh_g')
    output_3d = tf.reshape(output_2d, [-1, 100, 1], name='output_g')
    return output_3d


seq_length = 100
batch_size = 10
init_op = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.import_meta_graph('./model_c.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
sess.run(init_op)

graph = tf.get_default_graph()
Z = tf.placeholder(tf.float32, [None, None, 1])
output_3d = rebuild_model(graph, Z)
sess.run(output_3d, feed_dict={Z:sample_Z(10, 100, 1)})


# list all vars in checkpoint file
    # import tensorflow as tf
    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # latest_ckp = tf.train.latest_checkpoint('./')
    # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')