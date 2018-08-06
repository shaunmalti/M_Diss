import tensorflow as tf


with tf.Session() as sess:
  # Restore variables from disk.
  new_saver = tf.train.import_meta_graph('./model_test.ckpt.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))