import torch
import jax
from functools import partial

from utils.mlp2jax import torch_mlp2jax
from utils.resnet2jax import resnet18_jax_forward, extract_resnet_params_from_torch


def torch2jax(model: torch.nn.Module):
    if model.__class__.__name__ == "TinyTaxiNetDNN":
        fn = torch_mlp2jax(model)
        return jax.jit(fn)
    elif model.__class__.__name__ == "TaxiNetDNN":
        params = extract_resnet_params_from_torch(model, resnet_prefix='model')
        
        def forward_fn(x):
            return resnet18_jax_forward(params, x)

        return forward_fn
    else:
        raise NotImplementedError(f"Model type {model.__class__.__name__} not supported.")