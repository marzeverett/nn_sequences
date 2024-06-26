
experiments =  {
    #This is using Binary Cross Entropy and the full dataset 
    #This round got TOTALLY killed by bad naming in round 3, unfortunately 
    "run_1": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_2": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_3": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_4": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 64, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_5": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 64, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_6": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 100,
        "batch_size": 64, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_7": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_8": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_9": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 50,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_10": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 200,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_1",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_11": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 200,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_2",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
    "run_12": {
        "index_key": "time_steps",
        "consequent_key": "delta_frost_events",
        "epochs": 200,
        "batch_size": 32, 
        "sequence_limit": 240,
        "sequence_offset": 36,
        "model_name": "model_3",
        "verbose": False,
        "dataset_name": "240_36_delta",
    },
}