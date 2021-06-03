import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import DATA_FOLDER_PATH


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


def load_X_from_file(path, data_type):
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


def load_y_from_file(y_path):
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
    X_signals = load_X_from_file(path, data_type)
    return np.transpose(np.array(X_signals), (1, 2, 0))  # (totalStepNum*128*6)


def load_preprocess_y(y_path):
    y = load_y_from_file(y_path)
    y = y - 1  # Subtract 1 to each output class for friendly 0-based indexing
    y = one_hot(y)
    return y


def load_identification_data(dataset_number):
    """ data for identification is dataset 1-4
    """
    dataset_name = os.path.join(DATA_FOLDER_PATH, 'Dataset #' + str(dataset_number))

    X_train = load_preprocess_X(os.path.join(dataset_name, 'train', 'Inertial Signals'), data_type='train')
    X_test = load_preprocess_X(os.path.join(dataset_name, 'test', 'Inertial Signals'), data_type='test')

    train_label = load_preprocess_y(os.path.join(dataset_name, 'train', 'y_train.txt'))
    test_label = load_preprocess_y(os.path.join(dataset_name, 'test', 'y_test.txt'))

    return (X_train, X_test), (train_label, test_label)


def get_dataset_path(dataset_number):
    dataset_name = os.path.join(DATA_FOLDER_PATH, 'Dataset #' + str(dataset_number))
    return dataset_name