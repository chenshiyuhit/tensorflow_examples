import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./tensorflow_examples/MNIST_data/", one_hot=True)  # 对label进行one-hot编码，如：标签4表示为[0,0,0,0,1,0,0,0,0,0]，与神经网络输出层的格式对应

# hyperparameters
learning_rate = 0.5
epochs = 100
batch_size = 100

with tf.name_scope('Input'):
    # placeholder
    x = tf.placeholder(tf.float32, [None, 784], name="x") # input image 28*28
    y = tf.placeholder(tf.float32, [None, 10], name="labels")  # labels 0-9的one-hot编码

with tf.name_scope('Weights_Biases'):
    # hidden layer => w, b
    w1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')

    # output layer => w, b
    w2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

with tf.name_scope('Hidden_layer'):
    # hidden layer
    hidden_out = tf.add(tf.matmul(x, w1), b1)
    hidden_out = tf.nn.relu(hidden_out)

with tf.name_scope('Output_layer'):
    # output predict value
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

with tf.name_scope('Loss_function'):
    # loss function
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped) + (1-y) * tf.log(1-y_clipped), axis=1))

with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    # accuracy analysis
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init variables
init_op = tf.global_variables_initializer()

# add summary
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("w1", w1)
tf.summary.histogram("b1", b1)
tf.summary.histogram("w2", w2)
tf.summary.histogram("b2", b2)
summary_op = tf.summary.merge_all()

# begin the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./tensorflow_examples/graphs', graph=tf.get_default_graph())
    sess.run(init_op)   # init the variables
    total_batch = int(len(mnist.train.labels) / batch_size)

    for epoch in range(epochs):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})

        summary = sess.run(summary_op, feed_dict={x: batch_x, y:batch_y})
        writer.add_summary(summary, epoch)

        loss = sess.run(cross_entropy, feed_dict={x: batch_x, y:batch_y})
        print('Epoch:', (epoch+1), 'loss = {:.3f}' .format(loss))

    print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
