import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utils
from constants import PROJECT_PATH
from identification.CNN import CNNBatchNormLarge, CNN_BatchNorm, CNN
from identification.LSTMs import LSTMBatchNorm

# Training parameters
DATASET_NUMBER = 2
CHECKPOINT_PATH = os.path.join(PROJECT_PATH, 'code', "checkpoints/train")
PLOT_SAVE_PATH = os.path.join(PROJECT_PATH, 'plot', 'cnn_batchnorm_large')


class Callback(keras.callbacks.Callback):
    def __init__(self, monitor='acc', baseline=0.9):
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epochs, logs={}):
        if logs.get(self.monitor) >= self.baseline:
            print("\nReached 99.9% validation accuracy so cancelling training!")
            self.model.stop_training = True


def load_preprocess_X(X_signals):
    return np.transpose(np.array(X_signals), (1, 2, 0))  # (totalStepNum*128*6)


def load_preprocess_y(y):
    y = y - 1  # Subtract 1 to each output class for friendly 0-based indexing
    y = utils.one_hot(y)
    return y


def get_callbacks():
    return [
        Callback(monitor='val_acc', baseline=0.999),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, verbose=1, patience=20, restore_best_weights=True),
    ]


class Config:
    """ A class to store parameters,
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
        self.lambda_loss_amount = 0.0025
        self.training_epochs = 300
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 118  # Final output classes
        self.dropout = 0.2
        self.W = {
            'hidden': tf.Variable(tf.random.normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random.normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random.normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random.normal([self.n_classes]))
        }


class TwoStackedLSTMCells(keras.Model):
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
        self.lstm_cell_1 = layers.LSTMCell(self.config.n_hidden, dropout=config.dropout)
        self.lstm_cell_2 = layers.LSTMCell(self.config.n_hidden, dropout=config.dropout)
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


class CNN_Identification(keras.Model):
    def __init__(self):
        super().__init__()

        # shape after fc: (None, 64)
        self.model = keras.Sequential([
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


def last_full_connection_layer(config, lstm_output, cnn_output):
    x = layers.concatenate([lstm_output, cnn_output], 1)
    y = layers.Dense(config.n_classes, activation='softmax')(x)
    return y


# def experiment():
#     pass
#
#
# lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
#     0.001,
#     decay_steps=STEPS_PER_EPOCH*1000,
#     decay_rate=1,
#     staircase=False)
#
#
# def get_optimizer():
#   return keras.optimizers.Adam(lr_schedule)


# def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
#     if optimizer is None:
#         optimizer = get_optimizer()
#
#     model.compile(optimizer=optimizer,
#                   loss=keras.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=[
#                       keras.losses.BinaryCrossentropy(
#                           from_logits=True, name='binary_crossentropy'),
#                       'accuracy'])
#
#     model.summary()
#
#     history = model.fit(
#         train_ds,
#         steps_per_epoch=STEPS_PER_EPOCH,
#         epochs=max_epochs,
#         validation_data=validate_ds,
#         callbacks=get_callbacks(name),
#         verbose=0)
#     return history


if __name__ == '__main__':
    X = keras.Input(shape=[128, 6], dtype='float32')

    (X_train, X_test), (y_train, y_test) = utils.load_identification_data(dataset_number=DATASET_NUMBER,
                                                                          preprocess_x_func=load_preprocess_X,
                                                                          preprocess_y_func=load_preprocess_y)
    split_len = int(X_train.shape[0] * 0.3)
    X_val, y_val = X_train[-split_len:], y_train[-split_len:]
    X_train, y_train = X_train[:split_len], y_train[:split_len]

    config = Config(X_train, X_test)
    lstm_output = TwoStackedLSTMCells(config)(X)
    cnn_output = CNN_Identification()(X)
    pred_Y = last_full_connection_layer(config, lstm_output, cnn_output)

    opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model = keras.models.Model(inputs=X, outputs=pred_Y)
    print(model.summary())
    # model.save('../checkpoints/CNN_LSTM_weight.h5')
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics='acc')

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=config.batch_size,
                        epochs=config.training_epochs,
                        callbacks=get_callbacks()
                        )

    print("Evaluate on test data: ...")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    utils.plot_model_result(history, results, DATASET_NUMBER, config, saving_path=PLOT_SAVE_PATH)
