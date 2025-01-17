"""
    Code to train a DNN vision model to predict aircraft state variables
    REQUIRES:
        - raw camera images (training) in DATA_DIR + '/nominal_conditions'
        - validation data in DATA_DIR + '/nominal_conditions_val/'

    FUNCTIONALITY:
        - DNN is a ResNet-18 (pre-trained) with learnable final linear layer
            for regression with N=2 outputs
        - N=2 state outputs are:
            - distance_to_centerline_normalized
            - downtrack_position_NORMALIZED 
        - trains for configurable number of epochs
        - saves the best model params and loss plot in 
            - SCRATCH_DIR + '/DNN_train_taxinet/'

"""

import time
import copy
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

from model_taxinet import TaxiNetDNN, freeze_model, QuantTaxiNetDNN
from taxinet_dataloader import *
from plot_utils import *

from PIL import Image
from torchvision import transforms

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
model_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN_nick/'
debug_dir = NASA_ULI_ROOT_DIR + '/scratch/debug/'
# filename = "../../models/TinyTaxiNet.nnet"
# network = NNet(filename)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('found device: ', device)

model = TaxiNetDNN()

# load the pre-trained model
if device.type == 'cpu':
    model.load_state_dict(torch.load(model_dir + 'best_model.pt', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_dir + 'best_model.pt'))

model = model.to(device)
model.eval()

DATA_DIR = os.environ['NASA_ULI_DATA_DIR']

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *


def train_model(model, datasets, dataloaders, dist_fam, optimizer, device, results_dir, num_epochs=25, log_every=100):
    """
    Trains a model on datatsets['train'] using criterion(model(inputs), labels) as the loss.
    Returns the model with lowest loss on datasets['val']
    Puts model and inputs on device.
    Trains for num_epochs passes through both datasets.
    
    Writes tensorboard info to ./runs/ if given
    """
    writer = None
    writer = SummaryWriter(log_dir=results_dir)
        
    model = model.to(device)
    
    since = time.time()

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
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
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                running_tr_loss = 0.0 # used for logging

                running_n = 0
                
                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])
                for inputs, labels in dataloaders[phase]:
                    if phase == 'train':
                        n_tr_batches_seen += 1
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        # loss
                        ####################
                        loss = loss_func(outputs, labels).mean()
                        ####################
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

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

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
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
    plot_file = results_dir + '/loss.pdf' 
    basic_plot_ts(train_loss_vec, val_loss_vec, plot_file, legend = ['Train Loss', 'Val Loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__=='__main__':

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('found device: ', device)
    quantize = False
    
    # condition
    condition = 'morning'
    # larger images require a resnet, downsampled can have a small custom DNN
    dataset_type = 'large_images'

    model_name = 'resnet18'
    #model_name = 'squeezenet'


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

    # import pdb; pdb.set_trace()
    # where the training results should go
    fname = 'DNN_train_taxinet_' + str(model_name) + '_' + str(quantize)

    # results_dir = remove_and_create_dir(SCRATCH_DIR + fname + '/') 
    results_dir = '/home/nick/Documents/code/NASA_ULI_Xplane_Simulator/scratch/DNN_train_taxinet_resnet18_False/'

    # where raw images and csvs are saved
    BASE_DATALOADER_DIR = DATA_DIR + '/' + dataset_type  + '/' + condition

    train_dir = BASE_DATALOADER_DIR + '/' + condition + '_train'
    val_dir = BASE_DATALOADER_DIR + '/' + condition + '_validation'

    train_options = {"epochs": 5,
                     "learning_rate": 1e-3, 
                     "results_dir": results_dir,
                     "train_dir": train_dir, 
                     "val_dir": val_dir
                     }

    dataloader_params = {'batch_size': 512,
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
    optimizer = torch.optim.Adam(model.parameters(),
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

    # import pdb; pdb.set_trace()
    img = val_loader.dataset[2][0].detach().numpy().transpose([1, 2, 0])
    pil_img = Image.fromarray((img * 225).astype(np.uint8))
    pil_img.save(debug_dir+'dataloader_day2.png')

    dtpt_x = val_dataset[2][0].reshape((1,3,224,224))
    dtpt_y = val_dataset[2][1]

    model = model.cpu()
    import pdb; pdb.set_trace()
    y_pred = model(dtpt_x)
    print(loss_func(y_pred, dtpt_y))

    # img2 = Image.open(SCRATCH_DIR + 'debug/untransformed.png').convert('RGB')
    # img2_arr = np.asarray(img2)

    # tfms = transforms.Compose([transforms.Resize((224, 224)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406],
    #                                                          [0.229, 0.224, 0.225]),])
    # # tfms = transforms.Compose([transforms.Resize((width, height)), transforms.ToTensor()])
    # pil_img = Image.fromarray(img2_arr)
    # tfm_img = tfms(pil_img)

    # tfm_img2 = tfms(img2).detach().numpy().transpose([1, 2, 0])
    # pil_img2 = Image.fromarray((tfm_img2 * 225).astype(np.uint8))
    # pil_img2.save(debug_dir+'my_method_day2.png')
    tfms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]),])

    pil_img4 = Image.open(SCRATCH_DIR + 'debug/untransformed_day3.png').convert('RGB')
    pil_img4.save(debug_dir+'resave_ss_day3.png')
    tfm_img4 = tfms(pil_img4)
    img4 = tfm_img4.detach().numpy().transpose([1, 2, 0])
    pil_img42 = Image.fromarray((img4 * 225).astype(np.uint8))
    pil_img42.save(debug_dir+'my_method_day3.png')


    ### from dataset ####
    pil_img3 = Image.open(DATA_DIR + '/large_images/morning_nick/morning_nick_train/MWH_Runway04_morning_clear_11_304.png')
    pil_img3.save(debug_dir+'resave_dtpt_day3.png')
    tfm_img3 = tfms(pil_img3)
    img3 = tfm_img3.detach().numpy().transpose([1, 2, 0])
    pil_img32 = Image.fromarray((img3 * 225).astype(np.uint8))
    pil_img32.save(debug_dir+'my_method_from_data_day3.png')
    # for X_batch, y_batch in val_loader:
    #     y_pred = model(X_batch)
    #     loss = loss_func(y_pred, y_batch)
    #     print(loss)
    import pdb; pdb.set_trace()





    # train the DNN
    model = train_model(model, datasets, dataloaders, loss_func, optimizer, device, results_dir, num_epochs=train_options['epochs'], log_every=100)

    # save the best model to the directory
    torch.save(model.state_dict(), results_dir + "/best_model_nick.pt")

