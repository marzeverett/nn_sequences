import pandas as pd
import data.input_data.lgwf_specific_data as lgwf_specific_data
param_dict =  {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "sequence": True,
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": 17,
    }

train_input, train_output, test_input, test_output = lgwf_specific_data.get_train_and_test_data(param_dict)


