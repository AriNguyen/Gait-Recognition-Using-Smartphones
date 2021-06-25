import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utils

DATASET_NUMBER = 2
training_epochs = 200
batch_size = 512
learning_rate = 0.001


def preprocess_X(X_signals):
    X_signals = np.transpose(np.array(X_signals), (1, 0, 2))  # (totalStepNum*6*128)
    return X_signals.reshape(-1, 6, 128, 1)  # (totalStepNum*6*128*1)


def preprocess_y(y):
    # Substract 1 to each output class for friendly 0-based indexing
    y = y - 1
    # one_hot
    y = y.reshape(len(y))
    n_values = int(np.max(y)) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.model = tf.keras.Sequential([
            layers.Conv2D(32, (1, 9), strides=(1, 2), padding='SAME', activation='relu', input_shape=(128, 6, 1)),
            layers.MaxPool2D((1, 2), strides=(1, 2), padding='SAME'),
            layers.Conv2D(64, (1, 3), strides=(1, 1), padding='SAME', activation='relu'),
            layers.Conv2D(128, (1, 3), strides=(1, 1), padding='SAME', activation='relu'),
            layers.MaxPool2D((1, 2), strides=(1, 2), padding='VALID'),
            layers.Conv2D(128, (6, 1), strides=(1, 1), padding='VALID', activation='relu'),
            layers.Flatten(),
            layers.Dense(20, activation='softmax'),  # 20 or 119
        ])
        print(self.model.summary())

    def __call__(self, x):
        x = tf.reshape(x, [-1, 128, 6, 1])
        x = self.model(x)
        return x


class Callback(keras.callbacks.Callback):
    def __init__(self, monitor='acc', baseline=0.9):
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epochs, logs={}):
        if logs.get(self.monitor) >= self.baseline:
            print("\nReached 99.9% validation accuracy so cancelling training!")
            self.model.stop_training = True


def get_callbacks():
    return [
        Callback(monitor='acc', baseline=0.999),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, verbose=1, patience=20),
    ]


if __name__ == '__main__':
    X = tf.keras.Input(shape=[128, 6, 1], dtype='float32')

    (X_train, X_test), (y_train, y_test) = utils.load_identification_data(dataset_number=DATASET_NUMBER,
                                                                          preprocess_x_func=preprocess_X,
                                                                          preprocess_y_func=preprocess_y)

    cnn_output = CNN()(X)

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.models.Model(inputs=X, outputs=cnn_output)
    # model.save('../checkpoints/CNN_LSTM_weight.h5')
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics='acc')

    history = model.fit(X_train, y_train,
                        validation_split=0.3,
                        batch_size=batch_size,
                        epochs=training_epochs,
                        callbacks=get_callbacks(),
                        shuffle=True)

    print("Evaluate on test data: ...")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    # plot_model_result(history, results, DATASET_NUMBER, configNone, saving_path=PLOT_SAVE_PATH)
