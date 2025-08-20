import torch
import torch.nn as nn
from torchvision import models
import torchvision.models.quantization as quant_models
import torch.nn.functional as F


class TaxiNetDNN(nn.Module):
    def __init__(self, model_name="resnet18", quantize=False):
        super(TaxiNetDNN, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=True)
        elif model_name == 'squeezenet':
            self.model = models.squeezenet1_1(pretrained=True)

        if quantize:
            self.model = quant_models.resnet18(pretrained=True, quantize=True)

        y_dim = 2
        self.model.fc = nn.Linear(self.model.fc.in_features, y_dim)
        self.fc = self.model.fc

    def forward(self, z):
        out = self.model(z)
        return out


'''
EfficientNet model as CNN feature extractor
Takes in images and outputs embeddings
Input:
(N, C=3, H=224, W=224) = (batch size, channels, height, width)
Output:
(N, E=1280) = (batch size, feature number)
'''


class TaxiNetCNN(nn.Module):
    """
    Small CNN for regressing [cross_track_error, heading_error] from a 3x224x224 image.
    Ops: Conv -> ReLU -> Conv -> ReLU -> Flatten -> MLP (64->32->2).
    No pooling, no BN, no residuals (LiRPA-friendly).
    """
    def __init__(self, input_channels=3, H=224, W=224):
        super(TaxiNetCNN, self).__init__()
        print("Using TinyTaxiNetCNN")
        # Two lightweight conv layers; strides chosen to keep FC size small.
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=4, padding=0)  # -> 56x56
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0)              # -> 14x14

        # Compute flatten size analytically for the given H, W
        def out_dim(n, k, s, p, d=1):
            return (n + 2*p - d*(k - 1) - 1) // s + 1

        h1 = out_dim(H, 5, 4, 2)
        w1 = out_dim(W, 5, 4, 2)
        h2 = out_dim(h1, 3, 4, 1)
        w2 = out_dim(w1, 3, 4, 1)
        flatten_size = 32 * h2 * w2  # 32 * 14 * 14 = 6272 for 224x224

        # MLP head
        self.fc1 = nn.Linear(flatten_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # [cross_track_error, heading_error]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(z))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on output
        return x


def QuantTaxiNetDNN():
    # You will need the number of filters in the `fc` for future use.
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_fe = quant_models.resnet18(pretrained=True, progress=True, quantize=True)
    num_ftrs = model_fe.fc.in_features

    # Step 1. Isolate the feature extractor.
    model_fe_features = nn.Sequential(
    model_fe.quant,  # Quantize the input
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    model_fe.layer4,
    model_fe.avgpool,
    model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    new_head = nn.Sequential(
    nn.Linear(num_ftrs, 2),
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
    )

    return new_model


def freeze_model(model, freeze_frac=True):
    # freeze everything
    n_params = len(list(model.parameters()))
    for i, p in enumerate(model.parameters()):
        #if i < 6*n_params/7:
        if i < 4*n_params/7:
            p.requires_grad = False

    # make last layer trainable
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def unfreeze_model(model):
    global og_req_grads
    # unfreeze everything
    for p,v in zip( model.parameters(), og_req_grads):
        p.requires_grad = v
