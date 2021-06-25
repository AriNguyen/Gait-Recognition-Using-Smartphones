from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers


class CNNBatchNormLarge(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()

        # shape after fc: (None, 64)
        self.model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), strides=(1, 1), padding='SAME', activation='elu', input_shape=(128, 6, 1)),
            layers.BatchNormalization(),
            layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='SAME', activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), strides=(1, 1), padding='SAME', activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME'),
            layers.Flatten(),
            layers.Dense(64, activation='elu'),
        ])

    def __call__(self, x):
        x = tf.reshape(x, [-1, 128, 6, 1])
        x = self.model(x)
        return x