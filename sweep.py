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

# lstm_s
start = 20
end = 60
layer2 = 256
# lstm_s = 240

while( start <= end ):
    run_model(
            layer_lists = [[('lstm', int(start)), ('fc', int(layer2))]],

            root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_sweep/",
            transpose = [True],
            batch_size=90,
            loops=int(1000),
            train_percent=0.8,

            load_dataset = False,
            dataset_filename = "../../qcircml_code/data_07312023_sweep5/8/07312023_8_dataset.p", # filename of batched dataset
            show_plot = False,
            notes = "lstm_sweep: tanh activation on the lstm layer, lstm_size = " + str(start)
            # load = False,
            # model_load_filenames=["../../qcircml_code/data_07312023_sweep6/1/07312023_1_weights_final.h5"]
        )
    
    start += 4