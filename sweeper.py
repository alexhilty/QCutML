from rl_main import run_model
import datetime
import numpy as np

# file to set up multi dimensional sweeps (up to 2d for now)

######################### MODIFY THESE PARAMETERS #########################

# first dimension
sweep_arg = "layer_lists" # can be any argument that is passed to run_model
start = 10
end = 100
steps = 10

# second dimension, override with param list for arguments that are not numbers
sweep_arg2 = "None"
start2 = 64
end2 = 256
steps2 = 4

# make steps with numpy
values = np.linspace(start, end, steps)
values2 = np.linspace(start2, end2, steps2)

# override heere for non numeric arguments
values = [ [[ int(x), 80, [('fc', int(700)), ('fc', int(700/2))] ]] for x in values]
# values = [ [ [('flatten', None), ('fc', 1133), ('fc', 566)] ]] 
# values2 = [1, 2, 3]

# construct notes string based on sweep parameters
notes = "Testing multi gates with CORRECTED pointer network. Sweeping " + sweep_arg + " and " + sweep_arg2 + "."
sweep_num = 1 # number to append to data folder


######################### DO NOT MODIFY BELOW THIS LINE #########################

# construct parameter dicionary
param_dict = {
    "seed": 324, # seed for numpy and tensorflow

    "circ_filename": "../../qcircml_code/data/circol_base_4qubits_8gates_depth3_dict.p", # filename of circuit collection

    # batch parameters
    "load_dataset": False, # load dataset from file
    "dataset_filename": "../../qcircml_code/data_08152023_sweep2/0/08152023_0_dataset.p", # filename of batched dataset
    "batch_size": 90,
    "loops": int(100),
    "train_percent": 0.7,

    # model parameters
    "action_size": 8, # number of actions the agent can take
    "model_type": ["attention"], # type of model to use
    "layer_lists": [[ 80, 80, [('fc', int(700)), ('fc', int(700/2))] ]], # list of lists of number of hidden units for each desired fully connected layer (one list for each model)

    "learning_rate": 0.001, # learning rate for optimizer# define critic loss function
    "load": False, # load model weights from file
    "model_load_filenames": ["../../qcircml_code/data_07282023_2/07282023_14_weights.h5"], # filename of model weights, must be same size as layer_lists
    "transpose": [True], # whether to transpose the image before feeding it into the model, must be same size as layer_lists

    "validate_with_best": False, # validate with best checkpoint, FIXME: functionality broken with multiple models

    # training parameters
    "window_size": 100, # size of window for moving average
    "tf_function": True, # whether to use tf.function decorator

    # saving parameters
    "save": True, # save data to file
    "root_dir": "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep" + str(sweep_num) + "/",
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
        print("\n" + param_dict["notes"])
        param_dict[sweep_arg] = value

        # param_dict["loops"] = int(100 * 0.8 / value) #FIXME: comment this out later
        run_model(**param_dict)
    else:
        for value2 in values2:
            # modify notes
            param_dict["notes"] = notes + " " + sweep_arg + ": " + str(value) + ", " + sweep_arg2 + ": " + str(value2) + "."
            print("\n" + param_dict["notes"])
            param_dict[sweep_arg] = value
            param_dict[sweep_arg2] = value2
            run_model(**param_dict)