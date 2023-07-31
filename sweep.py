from rl_main import run_model
import datetime

# sweeping layer sizes, will keep three fully connected layers and increase their sizes

# start = 64
# end = 2048 * 2

# batch size
# start = 10
# end = 100
# step = 10
layer = 256

# while(start <= end):
#     fc_layer_list = [int(layer), int(layer / 2), int(layer / 4)]
#     run_model(
#         fc_layer_list = fc_layer_list,
#         batch_size = start,
#         loops = int(300 * start / 30),
#         train_percent = 0.8,
#         root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep5/",
#         load_dataset = False,
#         dataset_filename = "../../qcircml_code/data_07312023_sweep1/0/07312023_0_dataset.p", # filename of batched dataset
#         show_plot = False
#     )

#     start += step

run_model(
        fc_layer_list = [int(layer), int(layer / 2), int(layer / 4)],
        root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep5/",
        load_dataset = True,
        dataset_filename = "../../qcircml_code/data_07312023_sweep5/8/07312023_8_dataset.p", # filename of batched dataset
        show_plot = False,
        load = True,
        model_load_filename="../../qcircml_code/data_07312023_sweep5/11/07312023_11_weights_final.h5"
    )