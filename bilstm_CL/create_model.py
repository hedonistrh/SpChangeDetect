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

def create_model(featureplan):

	if (featureplan=="mfcc.txt"):
		frame_shape = (800, 40)
	elif (featureplan=="pyannote_based.txt"):
		frame_shape = (800, 59)
	else:
		print ("Incompatible featureplan")
		raise

	## Network Architecture

	input_frame = keras.Input(frame_shape, name='main_input')

	bidirectional_1 = layers.Bidirectional(layers.LSTM(128, activation="tanh", return_sequences=True))(input_frame)
	bidirectional_1_drop = layers.Dropout(0.2)(bidirectional_1)
	bidirectional_2 = layers.Bidirectional(layers.LSTM(80, activation='tanh', return_sequences=True))(bidirectional_1_drop)

	tdistributed_1 = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(bidirectional_2)
	tdistributed_1_drop = layers.Dropout(0.2)(tdistributed_1)
	tdistributed_2 = layers.TimeDistributed(layers.Dense(10, activation='tanh'))(tdistributed_1_drop)
	tdistributed_3 = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(tdistributed_2)



	model = Model(input_frame, tdistributed_3)

	rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0.0001, decay=0.00001)

	model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy"])

	return model
