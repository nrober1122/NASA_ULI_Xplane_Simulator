import copy
import os
import pathlib
import time

import h5py
import ipdb
import numpy as np
import torch
import tqdm
import wandb
from loguru import logger
from model_tiny_taxinet import TinyTaxiNetDNN
from torch.utils.data import DataLoader, TensorDataset

from utils.textfile_utils import remove_and_create_dir

NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]
DATA_DIR = pathlib.Path(os.environ["NASA_DATA_DIR"])
SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"


def get_dataloader(
    data_dir: pathlib.Path, stride: int, tts_name: str, dataloader_params
) -> tuple[TensorDataset, DataLoader]:
    label_file = data_dir / f"morning_downsampled_stride{stride}/morning_{tts_name}_stride{stride}.h5"
    f = h5py.File(label_file, "r")

    num_y = 2
    x_train = f["X_train"][()].astype(np.float32)
    y_train = f["y_train"][()].astype(np.float32)[:, 0:num_y]

    tensor_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))

    logger.info("Dataset size: {}".format(len(tensor_dataset)))
    logger.info("ymin: {}".format(tensor_dataset[:][1][:, 0].min()))
    logger.info("ymax: {}".format(tensor_dataset[:][1][:, 0].max()))
    logger.info("ymin: {}".format(tensor_dataset[:][1][:, 1].min()))
    logger.info("ymax: {}".format(tensor_dataset[:][1][:, 1].max()))
    tensor_dataloader = DataLoader(tensor_dataset, **dataloader_params)
    return tensor_dataset, tensor_dataloader


def train_model(
    model: TinyTaxiNetDNN,
    datasets: dict,
    dataloaders: dict,
    loss_func,
    optimizer,
    device,
    num_epochs=25,
    log_every=100,
):
    model = model.to(device)
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    tr_loss = np.nan
    val_loss = np.nan

    n_tr_batches_seen = 0

    train_loss_vec = []
    val_loss_vec = []

    since = time.time()
    with tqdm.tqdm(total=num_epochs, position=0) as pbar:
        pbar2 = tqdm.tqdm(total=dataset_sizes["train"], position=1)
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                running_tr_loss = 0.0  # used for logging

                running_n = 0

                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])
                for inputs, labels in dataloaders[phase]:
                    if phase == "train":
                        n_tr_batches_seen += 1

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)

                        # loss
                        ####################
                        loss = loss_func(outputs, labels).mean()
                        ####################

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.shape[0]

                    if phase == "train":
                        running_n += inputs.shape[0]
                        running_tr_loss += loss.item() * inputs.shape[0]

                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n

                            log_dict = {"Train/Loss": mean_loss}
                            wandb.log(log_dict, steep=n_tr_batches_seen)

                            running_tr_loss = 0.0
                            running_n = 0

                    pbar2.set_postfix(split=phase, batch_loss=loss.item())
                    pbar2.update(inputs.shape[0])

                epoch_loss = running_loss / dataset_sizes[phase]

                if phase == "train":
                    tr_loss = epoch_loss
                    train_loss_vec.append(tr_loss)

                if phase == "val":
                    val_loss = epoch_loss
                    # writer.add_scalar("loss/val", val_loss, n_tr_batches_seen)

                    log_dict = {"Val/Loss": val_loss}
                    wandb.log(log_dict, steep=n_tr_batches_seen)

                    val_loss_vec.append(val_loss)

                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss)

                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            pbar.update(1)
            pbar.set_description("loss tr: {:6.2e}, val: {:6.2e}".format(tr_loss, val_loss))

            # print(" ")
            # print("training loss: ", train_loss_vec)
            # print("val loss: ", val_loss_vec)
            # print(" ")

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # [1, 2, 4, 8]
    stride = 8

    train_options = {
        "epochs": 200,
        "learning_rate": 1e-3,
    }
    dataloader_params = {"batch_size": 256, "shuffle": True, "num_workers": 1, "drop_last": False, "pin_memory": True}

    width = 256 // stride
    height = 128 // stride
    n_features_in = width * height

    config = {"stride": stride, **train_options}
    wandb.init(project="tiny_taxinet_train", config=config)

    results_dir = remove_and_create_dir(SCRATCH_DIR + f"/tiny_taxinet_train_stride{stride}/")
    model = TinyTaxiNetDNN(n_features_in=n_features_in)

    train_dset, train_loader = get_dataloader(DATA_DIR, stride, "train", dataloader_params)
    val_dset, val_loader = get_dataloader(DATA_DIR, stride, "validation", dataloader_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_options["learning_rate"], amsgrad=True)

    loss_func = torch.nn.MSELoss().to(device)

    dsets = {"train": train_dset, "val": val_dset}
    dataloaders = {"train": train_loader, "val": val_loader}

    model = train_model(
        model,
        dsets,
        dataloaders,
        loss_func,
        optimizer,
        device,
        num_epochs=train_options["epochs"],
        log_every=100,
    )

    # save the best model to the directory
    torch.save(model.state_dict(), results_dir + "/best_model.pt")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
