import os
import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from utils import load_X_from_file, load_y_from_file, get_dataset_path


# Training parameters
BATCH_SIZE = 512
EPOCHS = 200
best_accuracy = 0


def load_preprocess_X(path, data_type):
    # X_signals = 6*totalStepNum*128
    X_signals = load_X_from_file(path, data_type)
    X_signals = np.transpose(np.array(X_signals), (1, 0, 2))  # (totalStepNum*6*128)
    return X_signals.reshape(-1, 6, 128, 1)  # (totalStepNum*6*128*1)


def load_preprocess_y(y_path):
    # Read dataset from disk, dealing with text file's syntax
    y = load_y_from_file(y_path)
    y = y - 1  # Subtract 1 to each output class for friendly 0-based indexing

    # one_hot
    y = y.reshape(len(y))
    n_values = int(np.max(y)) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS


class CNN_Network(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(32, (1, 9), strides=(1, 2), padding='SAME', activation='relu', input_shape=[6, 128, 1]),
            layers.MaxPool2D((1, 2), strides=(1, 2), padding='VALID'),
            layers.Conv2D(64, (1, 3), strides=(1, 1), padding='SAME', activation='relu'),
            layers.Conv2D(128, (1, 3), strides=(1, 1), padding='SAME', activation='relu'),
            layers.MaxPool2D((1, 2), strides=(1, 2), padding='VALID'),
            layers.Conv2D(128, (6, 1), strides=(1, 1), padding='VALID', activation='relu'),
            layers.Flatten(),
            layers.Dense(2, activation='softmax')
        ])
        print(self.model.summary())

    def __call__(self, x):
        # input shape [batch, in_height, in_width, in_channels]
        # kernel shape [filter_height, filter_width, in_channels, out_channels]
        y = self.model(x)
        return y


class Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('loss') < 0.01:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def get_callbacks(name=None):
    return [
        Callback(),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
    ]


if __name__ == '__main__':
    # Load and preprocess data
    dataset_path = get_dataset_path(dataset_number=5)
    X_train = load_preprocess_X(os.path.join(dataset_path, 'train', 'data'), data_type='train')
    X_test = load_preprocess_X(os.path.join(dataset_path, 'test', 'data'), data_type='test')

    y_train = load_preprocess_y(os.path.join(dataset_path, 'train', 'y_train.txt'))
    y_test = load_preprocess_y(os.path.join(dataset_path, 'test', 'y_test.txt'))

    # experiment
    X = keras.Input(dtype=tf.float32, shape=[6, 128, 1], name='cnn_X')
    label = keras.Input(dtype=tf.float32, shape=[98], name='cnn_Y')

    model = CNN_Network()
    h_fc = model(X)

    model = keras.models.Model(inputs=X, outputs=h_fc)
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['acc'],)

    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=get_callbacks())

    loss, acc = model.evaluate(X_test, y_test)
