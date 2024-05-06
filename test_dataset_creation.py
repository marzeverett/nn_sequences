import pandas as pd
import data.input_data.lgwf_specific_data as lgwf_specific_data
import models 
import run_experiments

param_dict =  {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 3,
        "batch_size": 32, 
        "sequence": True,
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "test_model",
        "verbose": True,
        "dataset_name": "240_36_delta",
        "experiment_name": "test_experiment_1"
    }

train_input, train_output, test_input, test_output = lgwf_specific_data.get_train_and_test_data(param_dict)
model = models.build_model(param_dict, train_input, train_output)

run_experiments.run_experiment(param_dict, model, train_input, train_output, test_input, test_output)