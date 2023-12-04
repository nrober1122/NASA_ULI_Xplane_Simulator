from nnet import *
from PIL import Image

from torchvision import transforms
import torch
import numpy as np
import time

import mss
import cv2
import os


from train_DNN.model_taxinet import TaxiNetDNN

# Read in the network
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
    model.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_dir + '/best_model.pt'))

model = model.to(device)
model.eval()


### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
width = 244    # Width of image
height = 244    # Height of image

screenShot = mss.mss()
monitor = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
screen_width = 360  # For cropping
screen_height = 200  # For cropping

def getCurrentImage():
    """ Returns a downsampled image of the current X-Plane 11 image
        compatible with the TinyTaxiNet neural network state estimator

        NOTE: this is designed for screens with 1920x1080 resolution
        operating X-Plane 11 in full screen mode - it will need to be adjusted
        for other resolutions
    """
    # Get current screenshot
    img = cv2.cvtColor(np.array(screenShot.grab(monitor)),
                       cv2.COLOR_BGRA2BGR)[230:, :, :]
    img = cv2.resize(img, (screen_width, screen_height))
    img_name = 'saving_from_run_ss.png'
    outDir = debug_dir
    # For now, just save the image to an output directory
    cv2.imwrite('%s%s' % (outDir, img_name), img)
    


    tfms = transforms.Compose([transforms.Resize((width, height)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]),])
    # tfms = transforms.Compose([transforms.Resize((width, height)), transforms.ToTensor()])
    pil_img3 = Image.open(outDir + img_name )
    tfm_img3 = tfms(pil_img3)
    img3 = tfm_img3.detach().numpy().transpose([1, 2, 0])
    pil_img32 = Image.fromarray((img3 * 225).astype(np.uint8))
    pil_img32.save(debug_dir+'my_method_from_data_simulated_day3.png')

    pil_img = Image.fromarray(img)
    tfm_img = tfms(pil_img)

    pil_img.save(debug_dir+'untransformed_day3.png')
    
    img3 = tfm_img.detach().numpy().transpose([1, 2, 0])
    pil_img32 = Image.fromarray((img3 * 225).astype(np.uint8))
    pil_img32.save(debug_dir+'direct_transformed_day3.png')
    # tfm_img.save(debug_dir+'transformed_day3.png')


    # import pdb; pdb.set_trace()


    # img = img[:, :, ::-1]
    # img = np.array(img)

    # # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so
    # # values range between 0 and 1
    # img = np.array(Image.fromarray(img).convert('L').crop(
    #     (55, 5, 360, 135)).resize((256, 128)))/255.0

    # # Downsample image
    # # Split image into stride x stride boxes, average numPix brightest pixels in that box
    # # As a result, img2 has one value for every box
    # img2 = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         img2[i, j] = np.mean(np.sort(
    #             img[stride*i:stride*(i+1), stride*j:stride*(j+1)].reshape(-1))[-numPix:])

    # # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
    # # The training data only contains images from sunny, 9am conditions.
    # # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
    # img2 -= img2.mean()
    # img2 += 0.5
    # img2[img2 > 1] = 1
    # img2[img2 < 0] = 0
    # import pdb; pdb.set_trace()
    # tfm_img = torch.transpose(tfm_img, 0, 2)
    # tfm_img = torch.transpose(tfm_img, 0, 1)
    return tfm_img3

def getStateDNN(client):
    """ Returns an estimate of the crosstrack error (meters)
        and heading error (degrees) by passing the current
        image through TinyTaxiNet

        Args:
            client: XPlane Client
    """
    image = getCurrentImage()
    # import pdb; pdb.set_trace()
    image = image.reshape(1, 3, width, height)
    # import pdb; pdb.set_trace()

    pred = model(image.to(device))
    # import pdb; pdb.set_trace()
    pred = pred.cpu().detach().numpy().flatten()
    # import pdb; pdb.set_trace()
    
    return pred[0]*10, pred[1]*30

