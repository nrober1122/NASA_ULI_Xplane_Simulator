import os
import pathlib

import time
import copy
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.colors import CenteredNorm
from torch.utils.data import DataLoader
import yaml

from simulation.controllers import get_p_control_torch, getProportionalControl
# from simulation.tiny_taxinet2 import get_network
# from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN
# from tiny_taxinet_train.tiny_taxinet_dataloader import tiny_taxinet_prepare_dataloader
from train_DNN.model_taxinet import TaxiNetCNN, TaxiNetDNN
from train_DNN.taxinet_dataloader import TaxiNetDataset

NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]
# DATA_DIR = os.environ["NASA_ULI_DATA_DIR"]

HJNNV_ROOT_DIR='/home/nick/code/hjnnv'
DATA_DIR = HJNNV_ROOT_DIR + '/data/scratch/NASA_ULI_Xplane_Simulator/'


class DNNAttackStatic(torch.nn.Module):
    def __init__(self, image_size: tuple[int, int, int], max_delta: float, rudder_target: float):
        super().__init__()
        self.patch = torch.nn.Parameter(torch.zeros(image_size, dtype=torch.float32))
        self.register_buffer("max_delta", torch.tensor(max_delta))
        self.register_buffer("rudder_target", torch.tensor(rudder_target))
        self.max_delta = self.max_delta
        self.rudder_target = self.rudder_target

    def clip_patch(self):
        self.patch.data = torch.clip(self.patch.data, -self.max_delta, self.max_delta)
        return self


# def train_model_atk(
#     model: TaxiNetDNN,
#     model_atk: DNNAttackStatic,
#     dl_train: DataLoader,
#     dl_test: DataLoader,
#     device: torch.device,
#     optimizer: torch.optim.Optimizer,
#     results_dir,
#     num_epochs=25,
#     log_every=100,
# ):
#     for epoch in range(num_epochs):
#         # Train.
#         model_atk.train()
#         loss_mses_train, loss_mses_eval = [], []
#         for inputs, _ in dl_train:
#             inputs = inputs.to(device)
#             batch_size = inputs.shape[0]

#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 # (b, 2)
#                 b_outputs = model(inputs + model_atk.patch)
#                 b_cte, b_he = b_outputs[:, 0], b_outputs[:, 1]
#                 b_rudder = get_p_control_torch(b_cte, b_he)
#                 assert b_rudder.shape == (batch_size,)

#                 # We want the rudder to be equal to the desired value.
#                 loss_mse = torch.mean((b_rudder - model_atk.rudder_target) ** 2)

#                 loss_mse.backward()
#                 optimizer.step()
#             loss_mses_train.append(loss_mse.detach().item())

#             # Project the gradient to within reasonable values.
#             model_atk = model_atk.clip_patch()

#         # Eval.
#         with torch.inference_mode():
#             model_atk.eval()
#             for inputs, _ in dl_test:
#                 inputs = inputs.to(device)
#                 batch_size = inputs.shape[0]

#                 b_outputs = model(inputs + model_atk.patch)
#                 b_cte, b_he = b_outputs[:, 0], b_outputs[:, 1]
#                 b_rudder = get_p_control_torch(b_cte, b_he)
#                 assert b_rudder.shape == (batch_size,)

#                 # We want the rudder to be equal to the desired value.
#                 loss_mse = torch.mean((b_rudder - model_atk.rudder_target) ** 2)
#                 loss_mses_eval.append(loss_mse.item())

#         # Log.
#         loss_mses_train = np.array(loss_mses_train)
#         loss_mses_eval = np.array(loss_mses_eval)
#         logger.info("Train: {:6.2e}, Test: {:6.2e}".format(loss_mses_train.mean(), loss_mses_eval.mean()))

#     return model_atk


def train_model_atk(
        model,
        model_atk,
        datasets,
        dataloaders,
        loss_func,
        optimizer,
        device,
        results_dir,
        num_epochs=25,
        log_every=100
    ):
    """
    Trains a model on datatsets['train'] using criterion(model(inputs), labels) as the loss.
    Returns the model with lowest loss on datasets['val']
    Puts model and inputs on device.
    Trains for num_epochs passes through both datasets.
    
    Writes tensorboard info to ./runs/ if given
    """
    writer = None
    writer = SummaryWriter(log_dir=results_dir)
        
    model_atk = model_atk.to(device)
    model_atk.train()
    
    since = time.time()

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model_atk.state_dict())
    best_loss = np.inf
    
    tr_loss = np.nan
    val_loss = np.nan
    
    n_tr_batches_seen = 0
   
    train_loss_vec = []
    val_loss_vec = []

    with tqdm(total=num_epochs, position=0) as pbar:
        pbar2 = tqdm(total=dataset_sizes['train'], position=1)
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_atk.train()  # Set model_atk to training mode
                else:
                    model_atk.eval()   # Set model_atk to evaluate mode

                running_loss = 0.0

                running_tr_loss = 0.0 # used for logging

                running_n = 0
                
                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])
                for inputs, labels in dataloaders[phase]:
                    batch_size = inputs.shape[0]
                    if phase == 'train':
                        n_tr_batches_seen += 1
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # loss
                        ####################
                        outputs_atk = model(inputs + model_atk.patch)
                        cte_atk, he_atk = outputs_atk[:, 0]*30, outputs_atk[:, 1]*10
                        rudder_atk = get_p_control_torch(cte_atk, he_atk)
                        assert rudder_atk.shape == (batch_size,)

                        # We want the rudder to be equal to the desired value.
                        # import pdb; pdb.set_trace()
                        loss = loss_func(rudder_atk, model_atk.rudder_target*(torch.ones((batch_size)).to(device))).mean()
                        ####################
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # Project the gradient to within reasonable values.
                        model_atk = model_atk.clip_patch()

                    # statistics
                    running_loss += loss.item() * inputs.shape[0]
                    
                    if phase =='train':
                        running_n += inputs.shape[0]
                        running_tr_loss += loss.item() * inputs.shape[0]
                        
                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n
                            
                            writer.add_scalar('loss/train', mean_loss, n_tr_batches_seen)
                            
                            running_tr_loss = 0.
                            running_n = 0
                    
                    pbar2.set_postfix(split=phase, batch_loss=loss.item())
                    pbar2.update(inputs.shape[0])
                    

                
                epoch_loss = running_loss / dataset_sizes[phase]
                
                if phase == 'train':
                    tr_loss = epoch_loss
                    train_loss_vec.append(tr_loss)

                if phase == 'val':
                    val_loss = epoch_loss
                    writer.add_scalar('loss/val', val_loss, n_tr_batches_seen)
                    val_loss_vec.append(val_loss)

                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss)

                # deep copy the model_atk
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model_atk.state_dict())
                    
            pbar.update(1)

            print(' ')
            print('training loss: ', train_loss_vec)
            print('val loss: ', val_loss_vec)
            print(' ')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    
    writer.flush()

    # plot the results to a file
    # plot_file = results_dir + '/loss.pdf' 
    # basic_plot_ts(train_loss_vec, val_loss_vec, plot_file, legend = ['Train Loss', 'Val Loss'])

    # load best model_atk weights
    model_atk.load_state_dict(best_model_wts)
    return model_atk


# def main():
#     scratch_dir = pathlib.Path(NASA_ULI_ROOT_DIR) / "scratch"
#     model_dir = pathlib.Path(NASA_ULI_ROOT_DIR) / "models/pretrained_DNN_nick"
#     results_dir = scratch_dir / "dnn_attack_static"
#     results_dir.mkdir(exist_ok=True, parents=True)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # dl_params = {"batch_size": 256, "shuffle": True, "num_workers": 1, "drop_last": False, "pin_memory": True}

#     model = TaxiNetDNN()

#     #load the pre-trained model
#     if device.type == 'cpu':
#         model.load_state_dict(torch.load(model_dir / 'best_model.pt', map_location=torch.device('cpu')))
#     else:
#         model.load_state_dict(torch.load(model_dir / 'best_model.pt'))
#     model.eval()
#     model.to(device)

#     # Freeze the network.
#     for param in model.parameters():
#         param.requires_grad = False

#     # Create the attack model.
#     image_size = (3, 224, 224)
#     max_delta = 0.027
#     rudder_target = 1.0
#     model_atk = DNNAttackStatic(image_size, max_delta, rudder_target)
#     model_atk.to(device)

#     # Create the dataloaders.
#     condition = 'morning'
#     # larger images require a resnet, downsampled can have a small custom DNN
#     dataset_type = 'large_images'

#     # model_name = 'resnet18'


#     # if quantize:
#     #     device = torch.device("cpu")
#     #     # MODEL
#     #     # instantiate the model and freeze all but penultimate layers
#     #     model = QuantTaxiNetDNN()
#     # else:
#     #     # MODEL
#     #     # instantiate the model and freeze all but penultimate layers
#     #     model = TaxiNetDNN(model_name=model_name, quantize=quantize)
#     #     model = freeze_model(model)

#     # import pdb; pdb.set_trace()
#     # where the training results should go
#     # fname = 'DNN_train_taxinet_' + str(model_name) + '_' + str(quantize)

#     # results_dir = remove_and_create_dir(SCRATCH_DIR + fname + '/') 
#     # results_dir = '/home/nick/Documents/code/NASA_ULI_Xplane_Simulator/scratch/DNN_train_taxinet_resnet18_False/'

#     # where raw images and csvs are saved
#     BASE_DATALOADER_DIR = DATA_DIR + '/' + dataset_type  + '/' + condition

#     train_dir = BASE_DATALOADER_DIR + '/' + condition + '_validation'
#     val_dir = BASE_DATALOADER_DIR + '/' + condition + '_test'

#     train_options = {"epochs": 5,
#                      "learning_rate": 1e-3, 
#                      "results_dir": results_dir,
#                      "train_dir": train_dir, 
#                      "val_dir": val_dir
#                      }

#     dataloader_params = {'batch_size': 512,
#                          'shuffle': True,
#                          'num_workers': 8,
#                          'drop_last': False}


#     # DATALOADERS
#     # instantiate the model and freeze all but penultimate layers
#     dset_train = TaxiNetDataset(train_options['train_dir'])
#     dset_test = TaxiNetDataset(train_options['val_dir'])

#     dl_train = DataLoader(dset_train, **dataloader_params)
#     dl_test = DataLoader(dset_test, **dataloader_params)


#     # OPTIMIZER
#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=train_options["learning_rate"],
#                                  amsgrad=True)

#     # LOSS FUNCTION
#     # loss_func = torch.nn.MSELoss().to(device)

#     # DATASET INFO
#     # datasets = {}
#     # datasets['train'] = train_dataset
#     # datasets['val'] = val_dataset

#     # dataloaders = {}
#     # dataloaders['train'] = train_loader
#     # dataloaders['val'] = val_loader




#     # cond_list = ["morning"]
#     # dset_train, dl_train = tiny_taxinet_prepare_dataloader(DATA_DIR, cond_list, "train", dl_params)
#     # dset_test, dl_test = tiny_taxinet_prepare_dataloader(DATA_DIR, cond_list, "validation", dl_params)
#     # optimizer = torch.optim.Adam(model_atk.parameters(), lr=5e-4, fused=True)

#     model_atk = train_model_atk(model, model_atk, dl_train, dl_test, device, optimizer, results_dir)

#     # Save the model.
#     torch.save(model_atk.state_dict(), results_dir / "model_atk.pt")

#     # Visualize the patch, the image and the patch + image.
#     # Also show the resulting state estimate and controls below the image.
#     image_raw = dset_test[0][0].cpu().numpy()
#     patch_np = model_atk.patch.detach().cpu().numpy()
#     assert patch_np.shape == image_raw.shape
#     image_atked = (image_raw + patch_np).clip(0, 1)

#     cte_raw, he_raw = model(torch.tensor(image_raw[None, :, None], device=device)).detach().cpu().numpy().squeeze()
#     rudder_raw = getProportionalControl(None, cte_raw, he_raw)

#     cte_atk, he_atk = model(torch.tensor(image_atked[None, :, None], device=device)).detach().cpu().numpy().squeeze()
#     rudder_atk = getProportionalControl(None, cte_atk, he_atk)

#     raw_descr = "CTE: {:.2f}, HE: {:.2f}\nRudder: {:.2f}".format(cte_raw, he_raw, rudder_raw)
#     atk_descr = "CTE: {:.2f}, HE: {:.2f}\nRudder: {:.2f}".format(cte_atk, he_atk, rudder_atk)

#     figsize = 2 * np.array([3 * 2, 1])
#     fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
#     im = axes[0].imshow(patch_np, norm=CenteredNorm(), cmap="RdBu_r")
#     fig.colorbar(im, ax=axes[0])
#     axes[1].imshow(image_raw, cmap="gray")
#     axes[2].imshow(image_atked, cmap="gray")
#     axes[0].set_title("Learned Patch")
#     axes[1].set_title("Raw Image\n{}".format(raw_descr))
#     axes[2].set_title("Perturbed Image\n{}".format(atk_descr))
#     [ax.axis("off") for ax in axes]
#     [ax.grid(False) for ax in axes]
#     fig.savefig(results_dir / "patch.pdf", bbox_inches="tight")

















# """
#     Code to train a DNN vision model to predict aircraft state variables
#     REQUIRES:
#         - raw camera images (training) in DATA_DIR + '/nominal_conditions'
#         - validation data in DATA_DIR + '/nominal_conditions_val/'

#     FUNCTIONALITY:
#         - DNN is a ResNet-18 (pre-trained) with learnable final linear layer
#             for regression with N=2 outputs
#         - N=2 state outputs are:
#             - distance_to_centerline_normalized
#             - downtrack_position_NORMALIZED 
#         - trains for configurable number of epochs
#         - saves the best model params and loss plot in 
#             - SCRATCH_DIR + '/DNN_train_taxinet/'

# """

# import time
# import copy
# import torch
# import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

# from model_taxinet import TaxiNetDNN, freeze_model, QuantTaxiNetDNN
# from taxinet_dataloader import *
# from plot_utils import *

# from PIL import Image
# from torchvision import transforms

# # make sure this is a system variable in your bashrc
# NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
# model_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN_nick/'
# debug_dir = NASA_ULI_ROOT_DIR + '/scratch/debug/'
# # filename = "../../models/TinyTaxiNet.nnet"
# # network = NNet(filename)

# torch.cuda.empty_cache()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('found device: ', device)

# DATA_DIR = os.environ['NASA_ULI_DATA_DIR']

# # where intermediate results are saved
# # never save this to the main git repo
# SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

# UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
# sys.path.append(UTILS_DIR)

# from textfile_utils import *


# def train_model(model, datasets, dataloaders, dist_fam, optimizer, device, results_dir, num_epochs=25, log_every=100):
#     """
#     Trains a model on datatsets['train'] using criterion(model(inputs), labels) as the loss.
#     Returns the model with lowest loss on datasets['val']
#     Puts model and inputs on device.
#     Trains for num_epochs passes through both datasets.
    
#     Writes tensorboard info to ./runs/ if given
#     """
#     writer = None
#     writer = SummaryWriter(log_dir=results_dir)
    
#     model = TaxiNetDNN()

#     # load the pre-trained model
#     if device.type == 'cpu':
#         model.load_state_dict(torch.load(model_dir + 'best_model.pt', map_location=torch.device('cpu')))
#     else:
#         model.load_state_dict(torch.load(model_dir + 'best_model.pt'))

#     model = model.to(device)
#     model.eval()

#     image_size = (3, 224, 224)
#     max_delta = 0.027
#     rudder_target = 1.0
#     model_atk = DNNAttackStatic(image_size, max_delta, rudder_target)


#     image_size = (3, 224, 224)
#     max_delta = 0.027
#     rudder_target = 1.0
#     model_atk = DNNAttackStatic(image_size, max_delta, rudder_target)
#     model_atk.to(device)
    
#     since = time.time()

#     dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

#     best_model_wts = copy.deepcopy(model_atk.state_dict())
#     best_loss = np.inf
    
#     tr_loss = np.nan
#     val_loss = np.nan
    
#     n_tr_batches_seen = 0
   
#     train_loss_vec = []
#     val_loss_vec = []

#     with tqdm(total=num_epochs, position=0) as pbar:
#         pbar2 = tqdm(total=dataset_sizes['train'], position=1)
#         for epoch in range(num_epochs):
#             # Each epoch has a training and validation phase
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model_atk.train()  # Set model to training mode
#                 else:
#                     model_atk.eval()   # Set model to evaluate mode

#                 running_loss = 0.0

#                 running_tr_loss = 0.0 # used for logging

#                 running_n = 0
                
#                 # Iterate over data.
#                 pbar2.refresh()
#                 pbar2.reset(total=dataset_sizes[phase])
#                 for inputs, labels in dataloaders[phase]:
#                     if phase == 'train':
#                         n_tr_batches_seen += 1
                    
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)

#                     # zero the parameter gradients
#                     optimizer.zero_grad()

#                     # forward
#                     # track history if only in train
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(inputs)

#                         # loss
#                         ####################
#                         loss = loss_func(outputs, labels).mean()
#                         ####################
                        
#                         # backward + optimize only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     # statistics
#                     running_loss += loss.item() * inputs.shape[0]
                    
#                     if phase =='train':
#                         running_n += inputs.shape[0]
#                         running_tr_loss += loss.item() * inputs.shape[0]
                        
#                         if n_tr_batches_seen % log_every == 0:
#                             mean_loss = running_tr_loss / running_n
                            
#                             writer.add_scalar('loss/train', mean_loss, n_tr_batches_seen)
                            
#                             running_tr_loss = 0.
#                             running_n = 0
                    
#                     pbar2.set_postfix(split=phase, batch_loss=loss.item())
#                     pbar2.update(inputs.shape[0])
                    

                
#                 epoch_loss = running_loss / dataset_sizes[phase]
                
#                 if phase == 'train':
#                     tr_loss = epoch_loss
#                     train_loss_vec.append(tr_loss)

#                 if phase == 'val':
#                     val_loss = epoch_loss
#                     writer.add_scalar('loss/val', val_loss, n_tr_batches_seen)
#                     val_loss_vec.append(val_loss)

#                 pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss)

#                 # deep copy the model
#                 if phase == 'val' and epoch_loss < best_loss:
#                     best_loss = epoch_loss
#                     best_model_wts = copy.deepcopy(model.state_dict())
                    
#             pbar.update(1)

#             print(' ')
#             print('training loss: ', train_loss_vec)
#             print('val loss: ', val_loss_vec)
#             print(' ')

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Loss: {:4f}'.format(best_loss))
    
#     writer.flush()

#     # plot the results to a file
#     plot_file = results_dir + '/loss.pdf' 
#     basic_plot_ts(train_loss_vec, val_loss_vec, plot_file, legend = ['Train Loss', 'Val Loss'])

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model


def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('found device: ', device)
    quantize = False
    
    # condition
    condition = 'morning'
    # larger images require a resnet, downsampled can have a small custom DNN
    dataset_type = 'images_3x224x224'

    model_name = 'resnet18'
    # model_name = 'squeezenet'

    # with open("simulation/config.yaml", "r") as f:
    #     config = yaml.safe_load(f)

    model_type = "cnn" # config["STATE_ESTIMATOR"]

    if model_type == "dnn":
        model_dir = NASA_ULI_ROOT_DIR + '/models/pretrained_DNN_nick/'
        model = TaxiNetDNN()
    elif model_type == "cnn":
        model_dir = NASA_ULI_ROOT_DIR + '/models/cnn_taxinet/'
        model = TaxiNetCNN()

    image_size = (3, 224, 224)
    max_delta = 0.03
    rudder_target = 1.0
    model_atk = DNNAttackStatic(image_size, max_delta, rudder_target)
    model_atk.to(device)

    # load the pre-trained model
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_dir + 'best_model.pt', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_dir + 'best_model.pt'))

    model = model.to(device)
    model.eval()


    # if quantize:
    #     device = torch.device("cpu")
    #     # MODEL
    #     # instantiate the model and freeze all but penultimate layers
    #     model = QuantTaxiNetDNN()
    # else:
    #     # MODEL
    #     # instantiate the model and freeze all but penultimate layers
    #     model = TaxiNetDNN(model_name=model_name, quantize=quantize)
    #     model = freeze_model(model)

    # where the training results should go
    fname = 'CNN_attack' + '_eps' + str(max_delta) + '_target' + str(rudder_target)

    # results_dir = remove_and_create_dir(SCRATCH_DIR + fname + '/') 
    results_dir = '/home/nick/code/hjnnv/data/scratch/attack_models/'  + fname + '/'

    # where raw images and csvs are saved
    BASE_DATALOADER_DIR = DATA_DIR + '/' + dataset_type  + '/' + condition

    train_dir = BASE_DATALOADER_DIR + '/' + condition + '_train'
    val_dir = BASE_DATALOADER_DIR + '/' + condition + '_validation'

    train_options = {"epochs": 3,
                     "learning_rate": 1e-3, 
                     "results_dir": results_dir,
                     "train_dir": train_dir, 
                     "val_dir": val_dir
                     }

    dataloader_params = {'batch_size': 128,
                         'shuffle': True,
                         'num_workers': 8,
                         'drop_last': False}


    # DATALOADERS
    # instantiate the model and freeze all but penultimate layers
    train_dataset = TaxiNetDataset(train_options['train_dir'])
    val_dataset = TaxiNetDataset(train_options['val_dir'])

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **dataloader_params)


    # OPTIMIZER
    optimizer = torch.optim.Adam(model_atk.parameters(),
                                 lr=train_options["learning_rate"],
                                 amsgrad=True)

    # LOSS FUNCTION
    loss_func = torch.nn.MSELoss().to(device)

    # DATASET INFO
    datasets = {}
    datasets['train'] = train_dataset
    datasets['val'] = val_dataset

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    # train the DNN
    model_atk = train_model_atk(model, model_atk, datasets, dataloaders, loss_func, optimizer, device, results_dir, num_epochs=train_options['epochs'], log_every=100)

    # save the best model to the directory
    # import pdb; pdb.set_trace()
    torch.save(model_atk.state_dict(), results_dir + "best_model.pt")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()