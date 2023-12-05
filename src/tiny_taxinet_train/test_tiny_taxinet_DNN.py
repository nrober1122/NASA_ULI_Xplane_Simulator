"""
    Tests a pre-trained DNN model
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm as tqdm

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

DATA_DIR = os.environ["NASA_DATA_DIR"]

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"

UTILS_DIR = NASA_ULI_ROOT_DIR + "/src/utils/"
sys.path.append(UTILS_DIR)


TRAIN_DNN_DIR = NASA_ULI_ROOT_DIR + "/src/train_DNN/"
sys.path.append(TRAIN_DNN_DIR)

from model_tiny_taxinet import TinyTaxiNetDNN
from test_final_DNN import test_model
from textfile_utils import *
from tiny_taxinet_dataloader import *

if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # base model dir, if you trained custom models
    # BASE_MODEL_DIR = SCRATCH_DIR + '/tiny_taxinet_DNN_train/' + train_str + '/'
    BASE_MODEL_DIR = NASA_ULI_ROOT_DIR + "/models/tiny_taxinet_pytorch/"

    print("found device: ", device)

    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + "/test_tiny_taxinet_DNN/")

    # evaluation_condition_list = ["afternoon", "morning", "overcast", "night"]
    evaluation_condition_list = ["morning"]

    # train_condition_list = [['afternoon'], ['morning'], ['overcast'], ['night'], ['afternoon', 'morning', 'overcast', 'night']]
    train_condition_list = [["morning"]]

    # larger images require a resnet, downsampled can have a small custom DNN
    dataset_type = "tiny_images"

    train_test_val = "test"

    dataloader_params = {"batch_size": 256, "shuffle": True, "num_workers": 1, "drop_last": False, "pin_memory": True}

    with open(results_dir + "/results.txt", "w") as f:
        header = "\t".join(
            [
                "Evaluation Condition",
                "Train Condition",
                "Train/Test",
                "Dataset Type",
                "Model Type",
                "Loss",
                "mean_inference_time_sec",
                "std_inference_time_sec",
            ]
        )
        f.write(header + "\n")

        for evaluation_condition in evaluation_condition_list:

            for train_condition in train_condition_list:

                train_str = "_".join(train_condition)

                if train_str == "afternoon_morning_overcast_night":
                    label_train_str = "All"
                else:
                    label_train_str = train_str

                model_dir = BASE_MODEL_DIR + "/" + train_str + "/"

                # load a model according to its training condition

                test_dataset, test_loader = tiny_taxinet_prepare_dataloader(
                    DATA_DIR, [evaluation_condition], "test", dataloader_params
                )

                # MODEL
                # instantiate the model
                model = TinyTaxiNetDNN()

                # load the pre-trained model
                if device.type == "cpu":
                    model.load_state_dict(torch.load(model_dir + "/best_model.pt", map_location=torch.device("cpu")))
                else:
                    model.load_state_dict(torch.load(model_dir + "/best_model.pt"))

                model = model.to(device)
                model.eval()

                # LOSS FUNCTION
                loss_func = torch.nn.MSELoss().to(device)

                # DATASET INFO
                datasets = {}
                datasets["test"] = test_dataset

                dataloaders = {}
                dataloaders["test"] = test_loader

                # TEST THE TRAINED DNN
                test_results = test_model(model, test_dataset, test_loader, device, loss_func)
                test_results["model_type"] = "trained"

                out_str = "\t".join(
                    [
                        evaluation_condition,
                        label_train_str,
                        train_test_val,
                        dataset_type,
                        test_results["model_type"],
                        str(test_results["losses"]),
                        str(test_results["time_per_item"]),
                        str(test_results["time_per_item_std"]),
                    ]
                )
                f.write(out_str + "\n")

                # COMPARE WITH A RANDOM, UNTRAINED DNN
                untrained_model = TinyTaxiNetDNN()
                untrained_model = untrained_model.to(device)

                test_results = test_model(untrained_model, test_dataset, test_loader, device, loss_func)
                test_results["model_type"] = "untrained"

                out_str = "\t".join(
                    [
                        evaluation_condition,
                        label_train_str,
                        train_test_val,
                        dataset_type,
                        test_results["model_type"],
                        str(test_results["losses"]),
                        str(test_results["time_per_item"]),
                        str(test_results["time_per_item_std"]),
                    ]
                )
                f.write(out_str + "\n")
