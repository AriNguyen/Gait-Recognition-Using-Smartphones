import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

from utils import load_identification_data, bias_variable, weight_variable

# Training parameters
batch_size = 512
epochs = 100
best_accuracy = 0
feature_shape = 20

dataset_number = 2


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
        self.n_classes = feature_shape  # Final output classes
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
        self.lstm_cell_1 = layers.LSTMCell(self.config.n_hidden)
        self.lstm_cell_2 = layers.LSTMCell(self.config.n_hidden)
        self.stacked_lstm = layers.StackedRNNCells([self.lstm_cell_1, self.lstm_cell_2] * self.config.n_layers)

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
        # rnn = layers.RNN(self.stacked_lstm, unroll=True, stateful=True)
        # state = rnn.get_initial_state(x)
        # outputs = rnn(x, training=True)
        outputs, states = tf.compat.v1.nn.static_rnn(self.stacked_lstm, x, dtype=tf.float32)

        # Get last time step's output feature for a "many to one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        return lstm_last_output


class CNN_Network(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # shape after fc: (None, 64)
        self.model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), strides=(1, 1), padding='SAME', activation='elu', input_shape=(128, 6, 1)),
            layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='SAME', activation='elu'),
            layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME'),
            layers.Flatten(),
            layers.Dense(64, activation='elu'),
        ])

    def __call__(self, x):
        x = tf.reshape(X, [-1, 128, 6, 1])
        x = self.model(x)
        return x


def last_full_connection_layer(lstm_output, cnn_output):
    eigen_input = tf.concat([lstm_output, cnn_output], 1)
    # 第四层，输入64维，输出118维，也就是具体的分类
    W_fc2 = weight_variable([128, feature_shape])
    b_fc2 = bias_variable([feature_shape])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数
    return tf.nn.softmax(tf.matmul(eigen_input, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, [None, 128, 6])  # 输入占位
    label = tf.compat.v1.placeholder(tf.float32, [None, feature_shape])  # label占位

    # Load dataset
    (X_train, X_test), (train_label, test_label) = load_identification_data(dataset_number=dataset_number)

    # Initialize
    config = Config(X_train, X_test)
    lstm_output = LSTM_Network(config)(X_train)
    cnn_output = CNN_Network()(X_train)
    pred_Y = last_full_connection_layer(lstm_output, cnn_output)

    cross_entropy = tf.math.reduce_mean(-tf.math.reduce_sum(label * tf.math.log(pred_Y + 1e-10), axis=[1]))  # 损失函数，交叉熵
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 使用adam优化
    correct_prediction = tf.math.equal(tf.argmax(pred_Y, 1), tf.argmax(label, 1))  # 计算准确度
    accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())  # 变量初始化

    for i in range(epochs):
        for start, end in zip(range(0, len(train_label), batch_size),
                              range(batch_size, len(train_label) + 1, batch_size)):
            sess.run(train_step, feed_dict={
                X: X_train[start:end],
                label: train_label[start:end]
            })

        # Test completely at every epoch: calculate accuracy
        accuracy_out, loss_out = sess.run(
            [accuracy, cross_entropy],
            feed_dict={
                X: X_test,
                label: test_label
            }
        )
        if accuracy_out > best_accuracy:
            best_accuracy = accuracy_out

        print(str(i) + 'th cross_entropy:', str(loss_out), 'accuracy:', str(accuracy_out))
    print("best accuracy:" + str(best_accuracy))
