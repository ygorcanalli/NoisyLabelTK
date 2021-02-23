import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization

def simple_mlp_32_64(input_shape, num_classes):
    model = Sequential(
        [
            InputLayer(input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(64,activation="relu"),
            Dense(num_classes, activation="softmax")
        ]
    )
    return model

# Taken from: https://keras.io/examples/vision/mnist_convnet/
def simple_conv_32_64(input_shape, num_classes):
    model = Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
