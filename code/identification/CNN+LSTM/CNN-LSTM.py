import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from utils import load_identification_data

dataset_number = 1

# Training parameters
batch_size = 512
epochs = 100
best_accuracy = 0
feature_shape = 118


class Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('loss') < 0.01:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


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
        self.lstm_cell_1 = layers.LSTMCell(self.config.n_hidden, dropout=0.5)
        self.lstm_cell_2 = layers.LSTMCell(self.config.n_hidden, dropout=0.5)
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
        x = tf.reshape(x, [-1, 128, 6, 1])
        x = self.model(x)
        return x


def last_full_connection_layer(lstm_output, cnn_output):
    x = layers.concatenate([lstm_output, cnn_output], 1)
    y = layers.Dense(feature_shape, activation='softmax')(x)
    return y


def experiment():
    pass


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    X = tf.keras.Input(shape=[128, 6], dtype='float32')
    label = tf.keras.Input(shape=[feature_shape], dtype='float32')

    # Load dataset
    (X_train, X_test), (y_train, y_test) = load_identification_data(dataset_number=dataset_number)

    # Reserve 10,000 samples for validation
    x_val = X_train[-10000:]
    y_val = y_train[-10000:]
    x_train = X_train[:-10000]
    y_train = y_train[:-10000]

    # Initialize
    config = Config(x_val, X_test)
    lstm_output = LSTM_Network(config)(X)
    cnn_output = CNN_Network()(X)
    pred_Y = last_full_connection_layer(lstm_output, cnn_output)

    checkpoint_path = "./checkpoints/train"
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    model = tf.keras.models.Model(inputs=X, outputs=pred_Y)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics='accuracy')

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[Callback(), earlyStopping])

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
