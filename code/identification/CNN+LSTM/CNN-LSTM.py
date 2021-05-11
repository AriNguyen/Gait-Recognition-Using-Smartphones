import tensorflow as tf
import numpy as np
import os

from constants import DATASET_1
from utils import one_hot, load_y


# load数据
def load_preprocess_X(path):
    X_signals = []
    files = os.listdir(path)
    for my_file in files:
        file_name = os.path.join(path, my_file)
        file = open(file_name, 'r')
        X_signals.append(
            [np.array(cell, dtype=np.float32) for cell in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()
        # X_signals = 6*totalStepNum*128
    return np.transpose(np.array(X_signals), (1, 2, 0))  # (totalStepNum*128*6)


def load_preprocess_y(y_path):
    y = load_y(y_path)

    # Subtract 1 to each output class for friendly 0-based indexing
    y = y - 1

    # one_hot
    y = one_hot(y)
    return y


# ---------------------------the part of CNN---------------------------------
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ----------------------------------the part of LSTM--------------------------------
class Config:
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 2  # nb of layers
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 118  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random.normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random.normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random.normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random.normal([self.n_classes]))
        }


class LSTM_Network(tf.keras.Model):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        self.lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.config.n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.config.n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([self.lstm_cell_1, self.lstm_cell_2] * self.config.n_layers, state_is_tuple=True)

    def __call__(self, x):
        # input shape: (batch_size, n_steps, n_input)
        x = tf.transpose(x, [1, 0, 2])  # permute n_steps and batch_size

        # Reshape to prepare input to hidden activation
        # new shape: n_steps * (batch_size, n_hidden)
        x = tf.reshape(x, [-1, self.config.n_inputs])

        # Linear activation
        x = tf.nn.relu(tf.matmul(x, self.config.W['hidden']) + self.config.biases['hidden'])

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        # new shape: n_steps * (batch_size, n_hidden)
        x = tf.split(x, self.config.n_steps, 0)

        # passing the concatenated vector to static_rnn
        outputs, states = tf.compat.v1.nn.static_rnn(self.lstm_cells, x, dtype=tf.float32)

        # Get last time step's output feature for a "many to one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        return lstm_last_output


class CNN_Network(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        CNN_input = tf.reshape(X, [-1, 128, 6, 1])

        w_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.elu(tf.nn.conv2d(CNN_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        w_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.elu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')

        w_fc1 = weight_variable([32 * 3 * 64, 64])
        b_fc1 = bias_variable([64])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 3 * 64])
        h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        return h_fc1


def last_full_connection_layer(lstm_output, cnn_output):
    eigen_input = tf.concat([lstm_output, cnn_output], 1)
    # 第四层，输入64维，输出118维，也就是具体的分类
    W_fc2 = weight_variable([128, 118])
    b_fc2 = bias_variable([118])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数
    return tf.nn.softmax(tf.matmul(eigen_input, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, [None, 128, 6])  # 输入占位
    label_ = tf.compat.v1.placeholder(tf.float32, [None, 118])  # label占位

    # 输入
    X_train = load_preprocess_X(os.path.join(DATASET_1, 'train/', 'Inertial Signals/'))
    X_test = load_preprocess_X(os.path.join(DATASET_1, 'test/', 'Inertial Signals/'))
    # 得到label
    train_label = load_preprocess_y(os.path.join(DATASET_1, 'train/', 'y_train.txt'))
    test_label = load_preprocess_y(os.path.join(DATASET_1, 'test/', 'y_test.txt'))

    config = Config(X_train, X_test)
    lstm_output = LSTM_Network(config)(X)
    cnn_output = CNN_Network()(X)
    pred_Y = last_full_connection_layer(lstm_output, cnn_output)

    cross_entropy = tf.math.reduce_mean(-tf.math.reduce_sum(label_ * tf.math.log(pred_Y + 1e-10), axis=[1]))  # 损失函数，交叉熵
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 使用adam优化
    correct_prediction = tf.math.equal(tf.argmax(pred_Y, 1), tf.argmax(label_, 1))  # 计算准确度
    accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())  # 变量初始化

    best_accuracy = 0
    for i in range(100):
        batch_size = 512
        for start, end in zip(range(0, len(train_label), batch_size),
                              range(batch_size, len(train_label) + 1, batch_size)):
            sess.run(train_step, feed_dict={
                X: X_train[start:end],
                label_: train_label[start:end]
            })
            # Test completely at every epoch: calculate accuracy
        accuracy_out, loss_out = sess.run(
            [accuracy, cross_entropy],
            feed_dict={
                X: X_test,
                label_: test_label
            }
        )
        if accuracy_out > best_accuracy:
            best_accuracy = accuracy_out
        print(str(i) + 'th cross_entropy:', str(loss_out), 'accuracy:', str(accuracy_out))

    print("best accuracy:" + str(best_accuracy))
