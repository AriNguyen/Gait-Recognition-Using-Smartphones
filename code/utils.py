import os
import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def one_hot(y):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y = y.reshape(len(y))
    n_values = int(np.max(y)) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS


def load_X(path: object, data_type: object) -> object:
    X_signals = []
    signal_types = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    for signal_type in signal_types:
        file_name = os.path.join(path, data_type + '_' + signal_type + '.txt')
        file = open(file_name, 'r')
        X_signals.append(
            [np.array(cell, dtype=np.float32) for cell in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()
        # X_signals = 6*totalStepNum*128
    return X_signals


def load_y(y_path):
    file = open(y_path, 'r')

    # Read dataset from disk, dealing with text file's syntax
    y = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )

    file.close()
    return y


def load_preprocess_X(path, data_type):
    # X_signals = 6*totalStepNum*128
    X_signals = load_X(path, data_type)
    return np.transpose(np.array(X_signals), (1, 2, 0))  # (totalStepNum*128*6)


def load_preprocess_y(y_path):
    y = load_y(y_path)

    # Subtract 1 to each output class for friendly 0-based indexing
    y = y - 1

    # one_hot
    y = one_hot(y)
    return y