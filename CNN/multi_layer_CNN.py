import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./tensorflow_examples/MNIST_data/", one_hot=True)  # 对label进行one-hot编码，如：标签4表示为[0,0,0,0,1,0,0,0,0,0]，与神经网络输出层的格式对应

# hyperparameters
learning_rate = 0.01
episodes = 2000
batch_size = 100

# Network parameters
num_input = 784
num_classes = 10
dropout = 0.75

# input placeholder
x = tf.placeholder(tf.float32, [None, num_input], name="x")
y = tf.placeholder(tf.float32, [None, num_classes], name="labels")

# dropout placeholder
keep_prob = tf.placeholder(tf.float32)

# layers weights and biases
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wf1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # output layer, 1024 inputs, 10 outputs
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bf1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# create some wrappers for simplicity
def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# CNN model
def conv_net(x, weights, biases, dropout):
    # reshape the input 784 features to match picture format [height*width*channel]
    # Tensor input become 4-D: [batch_size, height, width, channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv layer1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # maxpool layer1
    conv1 = maxpool2d(conv1)

    # conv layer2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # maxpool layer2
    conv2 = maxpool2d(conv2)

    # fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wf1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
    fc1 = tf.nn.relu(fc1)

    # dropput
    fc1 = tf.nn.dropout(fc1, dropout)

    # output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

# construct CNN model
logits = conv_net(x, weights, biases, keep_prob)
y_ = tf.nn.softmax(logits)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# evaluate model
correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init variables
init_op = tf.global_variables_initializer()

# tensorboard
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("wc1", weights['wc1'])
tf.summary.histogram("wc2", weights['wc2'])
tf.summary.histogram("wf1", weights['wf1'])
tf.summary.histogram("w_out", weights['out'])
tf.summary.histogram("bc1", biases['bc1'])
tf.summary.histogram("bc2", biases['bc2'])
tf.summary.histogram("bf1", biases['bf1'])
tf.summary.histogram("b_out", biases['out'])
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./tensorflow_examples/multi_layer_CNN_graphs', graph=tf.get_default_graph())

    # total_batch = int(len(mnist.train.labels) / batch_size)

    for episode in range(episodes):
        # for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        print('Episode: {}, loss: {}' .format(episode+1, loss_value))

        summary = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        writer.add_summary(summary, episode)

    print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: dropout}))