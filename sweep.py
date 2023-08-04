from rl_main import run_model
import datetime
import numpy as np

# sweeping layer sizes, will keep three fully connected layers and increase their sizes

start = 256
end = 2048 * 4

# batch size
# start = 10
# end = 100
# step = 10
layer = 512

# while(start <= end):
#     fc_layer_list = [int(start), int(start / 2), int(start / 4)]
#     run_model(
#         fc_layer_list = fc_layer_list,
#         batch_size = 90,
#         loops = 900,
#         train_percent = 0.8,
#         root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep6/",
#         load_dataset = True,
#         dataset_filename = "../../qcircml_code/data_07312023_sweep5/8/07312023_8_dataset.p", # filename of batched dataset
#         show_plot = False
#     )

#     start *= 2

# lstm_s
start = 0.001
end = 0.01
steps = 10
layer2 = 64
lstm_s = 24

# make steps with numpy
values = np.linspace(start, end, steps)

for value in values:
    lay_list = [[ ('lstm', int(lstm_s)), ('fc', int(layer2))]]
    for i in range(4):
        run_model(
                layer_lists = lay_list,

                root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep7/",
                transpose = [True],
                batch_size=90,
                loops=int(2000),
                train_percent=0.7,
                learning_rate=value,

                load_dataset = True,
                dataset_filename = "../../qcircml_code/data_08022023_sweep3/6/08022023_6_dataset.p", # filename of batched dataset
                show_plot = False,
                notes = "Optimizing learning rate for an lstm + fc model"
                # load = False,
                # model_load_filenames=["../../qcircml_code/data_07312023_sweep6/1/07312023_1_weights_final.h5"]
            )
        lay_list[0][1] = ('fc', int(layer2 * 2 ** (i + 1)))