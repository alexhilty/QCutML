from rl_main import run_model
import datetime
import numpy as np

# file to set up multi dimensional sweeps (up to 2d for now)


# first dimension
sweep_arg = "train_percent" # can be any argument that is passed to run_model
start = 0.5
end = 0.9
steps = 3

# second dimension, override with param list for arguments that are not numbers
sweep_arg2 = "None"
start2 = 64
end2 = 256
steps2 = 4

# make steps with numpy
values = np.linspace(start, end, steps)
values2 = np.linspace(start2, end2, steps2)

# construct notes string based on sweep parameters
notes = "Sweeping " + sweep_arg + " and " + sweep_arg2 + "."

# construct parameter dicionary
param_dict = {
    "seed": 324, # seed for numpy and tensorflow

    "circ_filename": "../../qcircml_code/data/circol_base_4qubits.p", # filename of circuit collection

    # batch parameters
    "load_dataset": False, # load dataset from file
    "dataset_filename": "../../qcircml_code/data_08022023_sweep3/6/08022023_6_dataset.p", # filename of batched dataset
    "batch_size": 90,
    "loops": int(2000),
    "train_percent": 0.7,

    # model parameters
    "action_size": 6, # number of actions the agent can take
    "layer_lists": [[('flatten', None), ('fc', 1024), ('fc', 256), ('fc', 128)]], # list of lists of number of hidden units for each desired fully connected layer (one list for each model)

    "learning_rate": 0.01, # learning rate for optimizer# define critic loss function
    "load": False, # load model weights from file
    "model_load_filenames": ["../../qcircml_code/data_07282023_2/07282023_14_weights.h5"], # filename of model weights, must be same size as layer_lists
    "transpose": [True], # whether to transpose the image before feeding it into the model, must be same size as layer_lists

    "validate_with_best": False, # validate with best checkpoint, FIXME: functionality broken with multiple models

    # training parameters
    "window_size": 100, # size of window for moving average

    # saving parameters
    "save": True, # save data to file
    "root_dir": "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep1/",
    "date_str": datetime.datetime.now().strftime("%m%d%Y"), # used for saving data

    # notes
    "notes": "",
    "show_plot": False
}

# perform sweep
for value in values:
    if sweep_arg2 == "None":
        # modify notes
        param_dict["notes"] = notes + " " + sweep_arg + ": " + str(value) + "."
        param_dict[sweep_arg] = value
        run_model(**param_dict)
    else:
        for value2 in values2:
            # modify notes
            param_dict["notes"] = notes + " " + sweep_arg + ": " + str(value) + ", " + sweep_arg2 + ": " + str(value2) + "."
            param_dict[sweep_arg] = value
            param_dict[sweep_arg2] = value2
            run_model(**param_dict)