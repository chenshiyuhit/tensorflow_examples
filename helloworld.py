import tensorflow as tf

helloworld = tf.constant('Hello world')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    hw = sess.run(helloworld)
    print(hw)
