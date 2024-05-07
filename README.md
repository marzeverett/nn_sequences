# nn_sequences
A repo for exploring sequence analysis with neural network based approaches. 

## Experiment Parameters Dict

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

* index_key: The name of the field used as the time index for sequence data 
* consequent_key: The (so far only binary supported) event of interest to predict
* epochs: Number of epochs for the network to run
* sequence_limit: Number of time steps to provide as input
* sequence_offset: Number of time steps ahead to predict
* model_name: Name of the NN model to be used
* verbose: Whether or not training should be verbose
* dataset_name: Name of the dataset to use
* experiment_name: Name of the experiment to run (Note -- if providing multiple experiments, the index of each is assumed to be the experiment name and added afterward.)


## TODO
* Run and see if F1 supported on other CPU 
* Run several experiments on differment models for frost prediction 
