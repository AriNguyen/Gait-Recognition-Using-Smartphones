"""Human activity recognition using smartphones dataset and an LSTM RNN."""

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

# The MIT License (MIT)
#
# Copyright (c) 2016 Guillaume Chevalier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Also thanks to Zhao Yu for converting the ".ipynb" notebook to this ".py"
# file which I continued to maintain.

# Note that the dataset must be already downloaded for this script to work.
# To download the dataset, do:
#     $ cd data/
#     $ python download_dataset.py

import numpy as np
import tensorflow as tf

from utils import one_hot, load_y
from constants import INPUT_SIGNAL_TYPES


# Load "X" (the neural network's training and testing inputs)
def load_preprocess_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)
def load_preprocess_y(y_path):
    y = load_y(y_path)

    # Subtract 1 to each output class for friendly 0-based indexing
    y = y - 1
    return y


class Config:
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 1  # nb of layers
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
        self.n_classes = 2  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random.normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random.normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random.normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random.normal([self.n_classes]))
        }


def LSTM_Network(X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
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
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    X = tf.transpose(X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    X = tf.reshape(X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    X = tf.nn.relu(tf.matmul(X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    X = tf.split(X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2] * config.n_layers,
                                                       state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cells, X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    DATA_PATH = "data/"
    DATASET_PATH = "data/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    TRAIN = "a/"
    TEST = "b/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "data/" + "train" + signal + '.txt' for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "data/" + "test" + signal + '.txt' for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_preprocess_X(X_train_signals_paths)
    X_test = load_preprocess_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    y_train = one_hot(load_preprocess_y(y_train_path))
    y_test = one_hot(load_preprocess_y(y_test_path))

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    X = tf.compat.v1.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.compat.v1.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.math.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.math.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.math.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    saver = tf.train.Saver(max_to_keep=1)

    sess = tf.compat.v1.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        if accuracy_out > best_accuracy:
            saver.save(sess, './lstm_ckpt/model')
            best_accuracy = accuracy_out

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    # ------------------------------------------------------------------
    # Step 5: Training is good, but having visual insight is even better
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook

    # ------------------------------------------------------------------
    # Step 6: And finally, the multi-class confusion matrix and metrics!
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook
