import os
import time

import cv2
import mss
import numpy as np
import torch
import jax

from train_DNN.model_taxinet import TaxiNetDNN
from utils.torch2jax import torch2jax

# Read in the network
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
model_dir = NASA_ULI_ROOT_DIR + '/models/pretrained_DNN_nick/'
debug_dir = NASA_ULI_ROOT_DIR + '/scratch/debug/'
# filename = "../../models/TinyTaxiNet.nnet"
# network = NNet(filename)

torch.cuda.empty_cache()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('found device: ', device)

model = TaxiNetDNN()

# load the pre-trained model
# using_torch = True
# if using_torch:
if device.type == 'cpu':
    model.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_dir + '/best_model.pt'))

model = model.to(device)
model.eval()

jax.config.update("jax_platform_name", "cpu")
jax_model = torch2jax(model)

import ipdb; ipdb.set_trace()