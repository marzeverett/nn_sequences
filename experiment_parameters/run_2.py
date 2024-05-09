
experiments =  {
    #This is using Binary Cross Entropy and the full dataset 
    #Difference from 1 is that it is using a 120/12 dataset
    "run_13": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_14": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_15": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_16": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 64, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_17": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 64, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_18": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 64, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_19": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_20": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_21": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_22": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 200,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_23": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 200,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
    "run_24": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 200,
        "batch_size": 32, 
        "sequence_limit": 120,
        "sequence_offset": 12,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "120_12_delta",
    },
}

