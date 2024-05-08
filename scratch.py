
import pickle
import os
import numpy as np 
import pandas as pd

# param_dict =  {
#         "index_key": "time_steps",
#         "consequent_key": "delta_frost_events",
#         "epochs": 3,
#         "batch_size": 32, 
#         "sequence_limit": 240,
#         "sequence_offset": 36,
#         "model_name": "test_model",
#         "verbose": True,
#         "dataset_name": "240_36_delta",
#         "experiment_name": "test_experiment_1"
#     }

# experiment_name = param_dict.get("experiment_name", "default")
# save_path = f"data/output_data/{experiment_name}/"
# predictions = np.load(f"{save_path}test_predictions.npy")
# test_output = np.load(f"{save_path}test_truth.npy")

# predictions = predictions.flatten()
# test_output = test_output.flatten()
# print(predictions.shape)
# print(test_output.shape)

# pred_dict = {"predictions": predictions, "truth": test_output}
# pred_df = pd.DataFrame(pred_dict)
# pred_df.to_csv(f"{save_path}pred_v_truth_test_set.csv")

features = ['bc:57:29:02:14:51_temperature', 'bc:57:29:02:14:51_humidity', 'bc:57:29:02:14:5f_temperature', 'bc:57:29:02:14:5f_humidity', 'bc:57:29:02:14:66_temperature', 'bc:57:29:02:14:66_humidity', 'bc:57:29:02:14:70_temperature', 'bc:57:29:02:14:70_humidity', 'bc:57:29:02:14:8b_temperature', 'bc:57:29:02:14:8b_humidity', 'bc:57:29:02:14:f5_temperature', 'bc:57:29:02:14:f5_humidity', 'bc:57:29:02:14:f6_temperature', 'bc:57:29:02:14:f6_humidity', 'bc:57:29:02:14:f8_temperature', 'bc:57:29:02:14:f8_humidity', 'AWR1_temperature', 'AWR1_humidity', 'AWR1_winddirection', 'AWR1_avewindspeed', 'AWR1_gustwindspeed', 'AWR1_light', 'AWR1_uv', 'AWR11_temperature', 'AWR11_humidity', 'AWR11_winddirection', 'AWR11_avewindspeed', 'AWR11_gustwindspeed', 'AWR11_light', 'AWR11_uv', 'AWR15_temperature', 'AWR15_humidity', 'AWR15_winddirection', 'AWR15_avewindspeed', 'AWR15_gustwindspeed', 'AWR15_light', 'AWR15_uv', 'AWR16_temperature', 'AWR16_humidity', 'AWR16_winddirection', 'AWR16_avewindspeed', 'AWR16_gustwindspeed', 'AWR16_light', 'AWR16_uv', 'AWR2_temperature', 'AWR2_humidity', 'AWR2_winddirection', 'AWR2_avewindspeed', 'AWR2_gustwindspeed', 'AWR2_light', 'AWR2_uv', 'AWR3_temperature', 'AWR3_humidity', 'AWR3_winddirection', 'AWR3_avewindspeed', 'AWR3_gustwindspeed', 'AWR3_light', 'AWR3_uv', 'AWR7_temperature', 'AWR7_humidity', 'AWR7_winddirection', 'AWR7_avewindspeed', 'AWR7_gustwindspeed', 'AWR7_light', 'AWR7_uv', 'AWR8_temperature', 'AWR8_humidity', 'AWR8_winddirection', 'AWR8_avewindspeed', 'AWR8_gustwindspeed', 'AWR8_light', 'AWR8_uv', 'AWR9_temperature', 'AWR9_humidity', 'AWR9_winddirection', 'AWR9_avewindspeed', 'AWR9_gustwindspeed', 'AWR9_light', 'AWR9_uv', 'WS1_temperature', 'WS1_humidity', 'WS1_winddirection', 'WS1_avewindspeed', 'WS1_gustwindspeed', 'WS1_light', 'WS1_uv', 'WS2_temperature', 'WS2_humidity', 'WS2_winddirection', 'WS2_avewindspeed', 'WS2_gustwindspeed', 'WS2_light', 'WS2_uv', 'WS3_temperature', 'WS3_humidity', 'WS3_winddirection', 'WS3_avewindspeed', 'WS3_gustwindspeed', 'WS3_light', 'WS3_uv', 'WS5_temperature', 'WS5_humidity', 'WS5_winddirection', 'WS5_avewindspeed', 'WS5_gustwindspeed', 'WS5_light', 'WS5_uv', 'WS6_temperature', 'WS6_humidity', 'WS6_winddirection', 'WS6_avewindspeed', 'WS6_gustwindspeed', 'WS6_light', 'WS6_uv']

print(len(features))


save_path = f"data/input_data/240_36_delta/"
train_input = np.load(f"{save_path}train_input.npy")
train_output = np.load(f"{save_path}train_output.npy")
test_input = np.load(f"{save_path}test_input.npy")
test_output = np.load(f"{save_path}test_output.npy")
print(train_input.shape)
print(train_output.shape)

print("Test")
index = 142

print(np.argmax(train_output>0.1))
# print(train_input[index][0])
print(train_output[index-50:index+50])
# print(test_input[1].shape)
# print(train_input[0][0])
# print(test_output[0])