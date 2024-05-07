
import pickle
import os
import numpy as np 
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
import tensorflow as tf
#import tensorflow_addons as tfa 
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.compat.v1.enable_eager_execution(
#     config=None, device_policy=None, execution_mode=None
# )
tf.config.experimental_run_functions_eagerly(True)
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


#I got help creating a custom metric from this article:
#https://medium.com/@balochdanish9980/deep-learning-writing-custom-loss-function-and-metric-function-in-tensorflow-806b6306603c 
#And from this stack overflow answer:
#https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras
#And here:
#https://github.com/tensorflow/tensorflow/issues/29799


loss_dict = {
    "mape": tf.keras.losses.MeanAbsolutePercentageError(),
    "mse": tf.keras.losses.MeanSquaredError(),
    "mae": tf.keras.losses.MeanAbsoluteError(),
    "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy(),
    "CategoricalCrossentropy": tf.keras.losses.CategoricalCrossentropy(),
    "BinaryAccuracy": tf.keras.metrics.BinaryAccuracy(),
    "Precision": tf.keras.metrics.Precision(),
    "Recall": tf.keras.metrics.Recall(),
    "TruePositives": tf.keras.metrics.TruePositives(),
    "TrueNegatives": tf.keras.metrics.TrueNegatives(),
    "FalsePositives": tf.keras.metrics.FalsePositives(),
    "FalseNegatives":  tf.keras.metrics.FalseNegatives(),
    "F1Score": tf.keras.metrics.F1Score()
}





@tf.function
def dice_loss(y_true, y_pred):
    #Function taken directly from here: 
    #https://github.com/tensorflow/tensorflow/issues/29799 
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 1 - (2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon()))

def compile_model(model):

    #Try this? 

    model.compile(
        optimizer="adam",
        loss= loss_dict['F1Score'],
        metrics=[loss_dict['BinaryAccuracy'], 
                loss_dict['TruePositives'],
                loss_dict['TrueNegatives'],
                loss_dict['FalsePositives'],
                loss_dict['FalseNegatives'],
                loss_dict['Precision'],
                loss_dict['Recall']]
    )
    print(model.summary())
    return model 

#The callbacks we will use 
def build_callbacks(param_dict):
    experiment_name = param_dict.get("experiment_name", "default")
    save_path = f"data/output_data/{experiment_name}/"
    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=True,
        monitor='val_loss',
        mode="min",
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        #Change is here!
        patience=10,
        mode="min",
        restore_best_weights=True,
    )
    return [save_best, early_stop]


def fit_model(param_dict, model, train_input, train_output):
    experiment_name = param_dict.get("experiment_name")
    #verbose=True
    callbacks = build_callbacks(param_dict)
    start_time = time.time()
    #Actually train it 
    history = model.fit(
        train_input, 
        train_output, 
        batch_size=param_dict.get("batch_size", 32), 
        epochs=param_dict.get("epochs", 50),
        verbose=param_dict.get("verbose",  False),
        callbacks=callbacks,
        validation_split=0.2,
        shuffle=False)
    end_time = time.time()
    #Restore the best model
    save_path = f"data/output_data/{experiment_name}/"
    try:
        model.load_weights(save_path)
    except Exception as e:
        print("Issue loading model: ", e)
    total_time = end_time - start_time
    return history, total_time

def evaluate_model(model, test_input, test_output):
    final_metrics = model.evaluate(x=test_input, y=test_output)
    metrics_dict = {}
    #Dictionary workaround from here: https://github.com/keras-team/keras/issues/14045
    final_metrics = {name: final_metrics[val] for val, name in enumerate(model.metrics_names)}
    f1_score = 2*(final_metrics["precision"] * final_metrics["recall"]) / (final_metrics["precision"] + final_metrics["recall"] + 0.0001)
    final_metrics['f1_score'] = f1_score
    return final_metrics

def save_model(model, param_dict):
    experiment_name = param_dict.get("experiment_name", "default")
    save_path = f"data/output_data/{experiment_name}/model"
    model.save(save_path)

def get_model_predictions(model, test_input):
    predictions = model.predict(test_input, verbose=False)
    return predictions 

def print_model_info(final_metrics):
    print("Final Metrics")
    print(final_metrics) 

def save_model_info(param_dict, predictions, test_output, final_metrics):
    experiment_name = param_dict.get("experiment_name", "default")
    save_path = f"data/output_data/{experiment_name}/"
    np.save(f"{save_path}test_predictions", predictions)
    np.save(f"{save_path}test_truth", test_output)
    df = pd.DataFrame(final_metrics, index=[0])
    df.to_csv(f"{save_path}final_metrics.csv")

def run_experiment(param_dict, model, train_input, train_output, test_input, test_output):
    #First compile the model 
    model = compile_model(model)
    #Run the training phase
    history, total_time = fit_model(param_dict, model, train_input, train_output)
    #Save the model 
    save_model(model, param_dict)
    print(f"Took {total_time} to run the model")
    #Next, evaluate the model 
    final_metrics = evaluate_model(model, test_input, test_output)
    #Get the predictions 
    predictions = get_model_predictions(model, test_input)
    #Save the info 
    save_model_info(param_dict, predictions, test_output, final_metrics)

    


# @tf.function
# def f1_score(y_true, y_pred):
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
#     ALL_PRED_POS = K.sum(y_pred)
#     ALL_TRUE_POS = K.sum(y_true)
#     TP = K.sum(y_true * y_pred)
#     all_pred = K.sum(y_true + 1) - ALL_TRUE_POS 
#     FP = ALL_PRED_POS - TP
#     true_zeros = K.sum(y_true + y_pred) 
#     TN = all_pred - true_zeros - TP 
#     FN = all_pred - TN - TP - FP 
#     Recall = (TP / TP + FN)
#     Precision = (TP / TP + FP)
#     #return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
#     return 2 * (Precision * Recall)+ K.epsilon() / (Precision + Recall + K.epsilon())


# @tf.function
# def f1_score(y_true, y_pred):
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
#     p = tf.keras.metrics.Precision()
#     p.update_state(y_true, y_pred)
#     precision = p.result()
#     #print("Precision")
#     #print(precision)
#     r = tf.keras.metrics.Recall()
#     r.update_state(y_true, y_pred)
#     recall = r.result()
#     #print("Recall")
#     #print(recall)
#     f1_score = 2*(precision * recall) / (precision + recall + K.epsilon())
#     #print("F1 score")
#     #print(f1_score)
#     return f1_score 
