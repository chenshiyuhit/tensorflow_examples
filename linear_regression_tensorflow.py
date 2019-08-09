import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# parameters
w = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

lr = 0.01
episodes = 1000

# predict model
y_ = tf.add(tf.multiply(w, X), b)

# loss function
loss = tf.reduce_sum(tf.pow(y_ - Y, 2)) / (2*n_samples)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# init variables
init_op = tf.global_variables_initializer()

# tensorboard
tf.summary.scalar("loss", loss)
tf.summary.scalar("w", w)
tf.summary.scalar("b", b)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())

    for episode in range(episodes):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        loss_value, w_value, b_value = sess.run([loss, w, b], feed_dict={X: train_X, Y: train_Y})
        print('Episode: {}, loss: {:.6f}, w: {}, b: {}' .format(episode+1, loss_value, w_value, b_value))

        summary = sess.run(summary_op, feed_dict={X: train_X, Y: train_Y})
        writer.add_summary(summary, episode)

    plt.plot(train_X, train_Y, 'ro', label='points')
    plt.plot(train_X, w_value*train_X + b_value, label='fitted line')
    plt.legend()
    plt.show()