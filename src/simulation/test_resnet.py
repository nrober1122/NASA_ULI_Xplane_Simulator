
# resnet_torch_to_jax.py
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
from torchvision.models import resnet18

import pretrained_dnn
from utils.resnet2jax import resnet18_jax_forward, extract_resnet_params_from_torch
from utils.torch2jax import torch2jax

# -------------------------
# Example usage snippet
# -------------------------
if __name__ == '__main__':
    # ---- Example: load your PyTorch TaxiNetDNN model (replace with your checkpoint) ----
    pt_resnet = pretrained_dnn.get_network()
    jax_resnet = torch2jax(pt_resnet)
    

    # For demonstration create a torchvision ResNet18 (weights random)
    
    # pt_resnet = resnet18(pretrained=False)
    # pt_resnet.eval()

    # Extract params
    # params = extract_resnet_params_from_torch(pt_resnet, resnet_prefix='model')

    # Create random input: NCHW
    rng = np.random.RandomState(0)
    x_torch = torch.tensor(rng.randn(1, 3, 224, 224).astype(np.float32))
    x_jax = jnp.array(x_torch.numpy())

    # PyTorch forward (to compare)
    with torch.no_grad():
        out_torch = pt_resnet(x_torch).numpy()

    # JAX forward
    # out_jax = resnet18_jax_forward(params, x_jax)
    out_jax = jax_resnet(x_jax)
    print("PyTorch output shape:", out_torch.shape)
    print("JAX output shape:", out_jax.shape)
    # You can compare numerically (they may differ if any eps/affine differences exist)
    print("Max abs diff example:", float(np.max(np.abs(out_jax - out_torch))))