
import pickle
import os
import numpy as np 
import pandas as pd

param_dict =  {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 3,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "test_model",
        "verbose": True,
        "dataset_name": "240_36_delta",
        "experiment_name": "test_experiment_1"
    }

experiment_name = param_dict.get("experiment_name", "default")
save_path = f"data/output_data/{experiment_name}/"
predictions = np.load(f"{save_path}test_predictions.npy")
test_output = np.load(f"{save_path}test_truth.npy")

predictions = predictions.flatten()
test_output = test_output.flatten()
print(predictions.shape)
print(test_output.shape)

pred_dict = {"predictions": predictions, "truth": test_output}
pred_df = pd.DataFrame(pred_dict)
pred_df.to_csv(f"{save_path}pred_v_truth_test_set.csv")
