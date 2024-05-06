

import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import datasets, layers, models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


def add_input_layer(model, x):
    dimensions = x.ndim
    #Num samples, num samples in sequnece, num features 
    if dimensions == 3:
        dims = (x.shape[1], x.shape[2])
        model.add(layers.Input(shape=dims))
    return model 

def add_output_layer(model, y):
    activation = "relu"
    num_dimensions = y.ndim
    print("Y shape")
    print(y.shape)
    if num_dimensions <= 2:
        features = 1 #Might want to update later 
        model.add(layers.Dense(features,
            activation=activation))
    return model 


def build_specific_model(model_name, model):
    if model_name == "test_model":
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu',use_bias=True))

    elif model_name == "model_1":
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(96, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu',use_bias=True))
    return model 


def build_model(param_dict, train_input, train_output):
    model_name = param_dict.get("model_name", "default")
    #Create the intial model 
    model = models.Sequential()
    #Build the input layer first
    model = add_input_layer(model, train_input) 
    #Then add the layers relevant for the specific model 
    model = build_specific_model(model_name, model)
    #Then add the output layer 
    model = add_output_layer(model, train_output)
    return model 


