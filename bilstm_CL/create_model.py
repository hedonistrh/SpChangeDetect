import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import keras

from keras import layers
from keras import models
from keras import optimizers
from keras.models import Model
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
import time

def create_model():

    frame_shape = (800, 40, 1)

    input_frame = keras.Input(frame_shape, name='main_input')

    conv1 = layers.Conv2D(50, kernel_size=(4, 4), strides=(1, 1), padding="same",
                        kernel_initializer="TruncatedNormal",
                        bias_initializer="TruncatedNormal")(input_frame)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    conv1_drop = layers.Dropout(0.15)(conv1)
    conv1_BN = layers.BatchNormalization()(conv1_drop)


    conv2 = layers.Conv2D(200, kernel_size=(3, 3), strides=(1, 1), padding="same",
                        kernel_initializer="TruncatedNormal",
                        bias_initializer="TruncatedNormal")(conv1_BN)
    conv2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    conv2_drop = layers.Dropout(0.15)(conv2)

    conv2_BN = layers.BatchNormalization()(conv2_drop)


    conv3 = layers.Conv2D(400, kernel_size=(4, 4), strides=(1, 1), padding="same",
                        kernel_initializer="TruncatedNormal",
                        bias_initializer="TruncatedNormal")(conv1)
    conv3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    conv3_drop = layers.Dropout(0.15)(conv3)
    conv3_BN = layers.BatchNormalization()(conv3_drop)


    xx = layers.TimeDistributed(layers.Flatten())(conv3_BN)


    tdistributed_1 = layers.TimeDistributed(layers.Dense(2000, kernel_initializer='TruncatedNormal',
                    bias_initializer='TruncatedNormal'))(xx)
    tdistributed_1 = layers.LeakyReLU()(tdistributed_1)
    tdistributed_1_BN = layers.BatchNormalization()(tdistributed_1)
    tdistributed_1_drop = layers.Dropout(0.15)(tdistributed_1_BN)

    tdistributed_2 = layers.TimeDistributed(layers.Dense(1, activation='sigmoid', 
                                                kernel_initializer='TruncatedNormal',
                                                bias_initializer='TruncatedNormal'))(tdistributed_1_drop)

    model = Model(input_frame, tdistributed_2)

    Nadam = keras.optimizers.Nadam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=0.00001, schedule_decay=0.0002)

    model.compile(loss='binary_crossentropy', optimizer="Nadam", metrics=['accuracy'])

    return model