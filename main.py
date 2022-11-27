


from utils.configuration import get_config_from_json
from utils.training_utilities import set_GPU
from utils.plotting_traces import plot_traces
from seq2point.seq2point import SEQ2POINT
from refit_loader.data_loader import REFIT_Loader
from dataset_management.dataloader import Seq2PointDataLoader
import builtins
import os
import torch
from pprint import pprint

builtins.MODEL_CONFIG = get_config_from_json(description="Model Parameters", config_file="configs/model_config.json")
builtins.DATASET_CONFIG = get_config_from_json(description="Dataset Management", config_file="configs/dataset_config.json")
builtins.TRAINING_CONFIG = get_config_from_json(description="Training Configuration", config_file="configs/training_config.json")
builtins.PLOT_CONFIG = get_config_from_json(description="Plot Settings", config_file="configs/plot_config.json")

dataloaders = Seq2PointDataLoader(target_appliance='kettle', target_houses= {'TRAIN' : [2], 'VALIDATE': [2], 'TEST':[2]}, proportion= {'train_percent':0.6, 'validate_percent':0.2}, subset_days=1)

network = SEQ2POINT().to(set_GPU())

results = network.run(dataloaders.train_dataloader, dataloaders.validation_dataloader, assess_training=False)

plot_traces(traces = [results[0], results[1]], labels=['training', 'validation'], axis_labels=['Epochs', 'Loss'], title='House 2 Training MSE Loss vs Validation MSE Loss per Epoch')