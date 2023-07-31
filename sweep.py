from rl_main import run_model
import datetime

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

# training percent
start = 0.2
end = 0.9

while( start <= end ):
    run_model(
            fc_layer_list = [int(layer), int(layer / 2), int(layer / 4)],
            root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep7/",
            batch_size=90,
            loops=int(1800 * 0.8/start),
            train_percent=start,
            load_dataset = False,
            dataset_filename = "../../qcircml_code/data_07312023_sweep5/8/07312023_8_dataset.p", # filename of batched dataset
            show_plot = False,
            load = False,
            model_load_filename="../../qcircml_code/data_07312023_sweep6/1/07312023_1_weights_final.h5"
        )
    
    start += 0.1