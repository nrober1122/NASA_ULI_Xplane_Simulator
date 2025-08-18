# resnet_torch_to_jax.py
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
import pretrained_dnn

jax.config.update("jax_platform_name", "cpu")

# -------------------------
# Helpers for conv / pooling
# -------------------------
def _to_padding_tuple(p):
    """Convert PyTorch padding (int or tuple) to ((pad_top,pad_bottom),(pad_left,pad_right))
       We'll return a padding list suitable for lax.conv_general_dilated's `padding` arg
       for spatial dims only (we'll wrap later for all dims).
    """
    if isinstance(p, int):
        return (p, p), (p, p)
    if len(p) == 2:
        return (p[0], p[0]), (p[1], p[1])
    # allow single-element tuple like (3,) -> same for both
    if len(p) == 1:
        return (p[0], p[0]), (p[0], p[0])
    raise ValueError("Unexpected padding format: %r" % (p,))


def conv2d_jax(x, w, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    x: jnp array NCHW
    w: jnp array OIHW (PyTorch conv weight layout)
    stride: (sh, sw)
    padding: (pad_h, pad_w) - symmetric
    dilation: (dh, dw)
    """
    # lax expects padding as sequence for each spatial dim like ((pad_top, pad_bottom), (pad_left, pad_right))
    (pad_h, pad_w) = padding
    padding_for_lax = ((pad_h, pad_h), (pad_w, pad_w))
    # dimension numbers: input NCHW, kernel OIHW, output NCHW
    out = lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=stride,
        padding=padding_for_lax,
        rhs_dilation=dilation,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )
    return out


def maxpool2d_jax(x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)):
    """
    Implements PyTorch MaxPool2d with symmetric padding and no ceil_mode.
    x: NCHW
    """
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    # reduce_window expects window dims for every dimension; for NCHW: (1,1,kh,kw)
    window_dims = (1, 1, kh, kw)
    strides = (1, 1, sh, sw)
    padding_lax = [(0, 0), (0, 0), (ph, ph), (pw, pw)]
    return lax.reduce_window(x, -jnp.inf, lax.max, window_dims, strides, padding_lax)


def avgpool2d_jax(x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)):
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    # apply symmetric padding first
    x = jnp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=0)
    # reshape hack: compute mean over each window using reduce_window_sum then divide
    window_dims = (1, 1, kh, kw)
    strides = (1, 1, sh, sw)
    padding_lax = [(0, 0), (0, 0), (0, 0), (0, 0)]  # we've already padded explicitly
    summed = lax.reduce_window(x, 0.0, lax.add, window_dims, strides, padding_lax)
    return summed / (kh * kw)


def adaptive_avg_pool_1x1(x):
    # x: NCHW -> average over H and W -> result shape N C 1 1
    return jnp.mean(x, axis=(2, 3), keepdims=True)


# -------------------------
# BatchNorm (inference mode)
# -------------------------
def batch_norm_eval(x, gamma, beta, running_mean, running_var, eps):
    """
    x: NCHW
    gamma, beta, running_mean, running_var: 1D arrays shaped (C,)
    """
    # reshape to broadcast NCHW
    mean = running_mean.reshape((1, -1, 1, 1))
    var = running_var.reshape((1, -1, 1, 1))
    gamma_r = gamma.reshape((1, -1, 1, 1))
    beta_r = beta.reshape((1, -1, 1, 1))
    return gamma_r * (x - mean) / jnp.sqrt(var + eps) + beta_r


# -------------------------
# BasicBlock (ResNet)
# -------------------------
def basic_block_jax(x, params, downsample=None, stride=(1, 1)):
    """
    params: dict with keys:
      - conv1_w, bn1_w (gamma), bn1_b (beta), bn1_rm, bn1_rv, bn1_eps
      - conv2_w, bn2_w, bn2_b, bn2_rm, bn2_rv, bn2_eps
    downsample: None or dict with conv_w and bn_{w,b,rm,rv,eps}
    stride: tuple used for conv1 stride
    """
    identity = x

    # conv1
    out = conv2d_jax(
        x,
        params['conv1_w'],
        stride=stride,
        padding=params.get('conv1_padding', (0, 0)),
        dilation=params.get('conv1_dilation', (1, 1)),
    )
    out = batch_norm_eval(out,
                          params['bn1_w'],
                          params['bn1_b'],
                          params['bn1_running_mean'],
                          params['bn1_running_var'],
                          params.get('bn1_eps', 1e-5))
    out = jnp.maximum(out, 0)

    # conv2
    out = conv2d_jax(
        out,
        params['conv2_w'],
        stride=(1, 1),
        padding=params.get('conv2_padding', (0, 0)),
        dilation=params.get('conv2_dilation', (1, 1)),
    )
    out = batch_norm_eval(out,
                          params['bn2_w'],
                          params['bn2_b'],
                          params['bn2_running_mean'],
                          params['bn2_running_var'],
                          params.get('bn2_eps', 1e-5))

    # downsample if provided
    if downsample is not None:
        identity = conv2d_jax(
            identity,
            downsample['conv_w'],
            stride=stride,
            padding=downsample.get('conv_padding', (0, 0)),
        )
        identity = batch_norm_eval(identity,
                                   downsample['bn_w'],
                                   downsample['bn_b'],
                                   downsample['bn_running_mean'],
                                   downsample['bn_running_var'],
                                   downsample.get('bn_eps', 1e-5))

    out = out + identity
    out = jnp.maximum(out, 0)
    return out


# -------------------------
# ResNet forward (functional)
# -------------------------
def resnet18_jax_forward(params, x):
    """params: dict produced by extract_resnet_params (see below)
       x: input as jnp array NCHW (float32)
    """
    # conv1
    out = conv2d_jax(
        x,
        params['conv1']['weight'],
        stride=params['conv1'].get('stride', (2, 2)),
        padding=params['conv1'].get('padding', (3, 3)),
        dilation=params['conv1'].get('dilation', (1, 1)),
    )
    out = batch_norm_eval(out,
                          params['bn1']['weight'],
                          params['bn1']['bias'],
                          params['bn1']['running_mean'],
                          params['bn1']['running_var'],
                          params['bn1'].get('eps', 1e-5))
    out = jnp.maximum(out, 0)
    out = maxpool2d_jax(out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    # out = avgpool2d_jax(out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    # layers: each layer is a list of blocks with block-specific params
    for li in range(1, 5):
        layer = params[f'layer{li}']  # a list of block dicts
        for block_idx, block in enumerate(layer):
            stride = block.get('stride', (1, 1))
            downsample = block.get('downsample', None)
            out = basic_block_jax(out, block['params'], downsample=downsample, stride=stride)

    # avgpool (adaptive to 1x1)
    out = adaptive_avg_pool_1x1(out)  # N C 1 1
    out = jnp.reshape(out, (out.shape[0], -1))  # N, C

    # fc
    fc_w = params['fc']['weight']  # shape (out_features, in_features) as in PyTorch
    fc_b = params['fc']['bias']    # shape (out_features,)
    # JAX dot: (N, in) @ (in, out).T or jnp.dot(x, fc_w.T)
    out = jnp.dot(out, fc_w.T) + fc_b
    return out


# -------------------------
# Parameter extraction & conversion
# -------------------------
def _make_key(prefix: str, name: str) -> str:
    """Return 'prefix.name' if prefix is non-empty else 'name' (no leading dot)."""
    return f"{prefix}.{name}" if prefix else name

def extract_resnet_params_from_torch(torch_model, resnet_prefix: str = 'model'):
    """
    Extract parameters from a PyTorch ResNet-like module into a params dict
    usable by the functional JAX ResNet forward (NCHW / OIHW conv layout).
    - torch_model: full PyTorch model (e.g. TaxiNetDNN or torchvision resnet)
    - resnet_prefix: attribute name of the ResNet inside torch_model, or ''
                     if the ResNet *is* the root model itself.
    """
    sd = torch_model.state_dict()

    def t2j(key):
        """Get key from state_dict and convert to jnp.array with helpful error on miss."""
        if key not in sd:
            raise KeyError(f"Missing key in state_dict: '{key}'. Available keys (sample): "
                           f"{list(sd.keys())[:10]} ...")
        return jnp.array(sd[key].cpu().numpy())

    # helper to resolve the python module that contains the ResNet
    if resnet_prefix:
        try:
            resnet_module = getattr(torch_model, resnet_prefix)
        except AttributeError:
            raise AttributeError(f"Model has no attribute '{resnet_prefix}'")
    else:
        resnet_module = torch_model

    params = {}

    # top conv + bn
    params['conv1'] = {
        'weight': t2j(_make_key(resnet_prefix, 'conv1.weight')),
        'stride': (2, 2),
        'padding': (3, 3),
        'dilation': (1, 1),
    }
    params['bn1'] = {
        'weight': t2j(_make_key(resnet_prefix, 'bn1.weight')),
        'bias': t2j(_make_key(resnet_prefix, 'bn1.bias')),
        'running_mean': t2j(_make_key(resnet_prefix, 'bn1.running_mean')),
        'running_var': t2j(_make_key(resnet_prefix, 'bn1.running_var')),
        'eps': float(getattr(getattr(resnet_module, 'bn1'), 'eps', 1e-5))
    }

    # iterate layers 1..4 using the actual modules (safer than parsing keys)
    for li in range(1, 5):
        layer_attr = f'layer{li}'
        if not hasattr(resnet_module, layer_attr):
            raise AttributeError(f"ResNet module is missing '{layer_attr}'")
        layer_module = getattr(resnet_module, layer_attr)
        layer_list = []

        # iterate over blocks using the layer_module (Sequential of BasicBlock)
        for idx, block_module in enumerate(layer_module):
            block_base = f'{_make_key(resnet_prefix, f"layer{li}")}.{idx}'
            # conv1 weight
            conv1_w = t2j(f'{block_base}.conv1.weight')
            # conv1 stride and padding come from module attributes
            conv1_stride = tuple(getattr(block_module.conv1, 'stride', (1, 1)))
            conv1_padding = tuple(getattr(block_module.conv1, 'padding', (0, 0)))
            conv1_dilation = tuple(getattr(block_module.conv1, 'dilation', (1, 1)))

            # bn1 params
            bn1_w = t2j(f'{block_base}.bn1.weight')
            bn1_b = t2j(f'{block_base}.bn1.bias')
            bn1_rm = t2j(f'{block_base}.bn1.running_mean')
            bn1_rv = t2j(f'{block_base}.bn1.running_var')
            bn1_eps = float(getattr(block_module.bn1, 'eps', 1e-5))

            # conv2 weight
            conv2_w = t2j(f'{block_base}.conv2.weight')
            conv2_padding = tuple(getattr(block_module.conv2, 'padding', (0, 0)))
            conv2_dilation = tuple(getattr(block_module.conv2, 'dilation', (1, 1)))

            # bn2 params
            bn2_w = t2j(f'{block_base}.bn2.weight')
            bn2_b = t2j(f'{block_base}.bn2.bias')
            bn2_rm = t2j(f'{block_base}.bn2.running_mean')
            bn2_rv = t2j(f'{block_base}.bn2.running_var')
            bn2_eps = float(getattr(block_module.bn2, 'eps', 1e-5))

            # downsample (optional)
            downsample = None
            if hasattr(block_module, 'downsample') and block_module.downsample is not None:
                # Expect downsample[0] = conv, downsample[1] = bn (typical BasicBlock)
                # Keys are e.g. '...downsample.0.weight' and '...downsample.1.weight'
                down_conv_key = f'{block_base}.downsample.0.weight'
                down_bn_key = f'{block_base}.downsample.1.weight'
                if down_conv_key in sd and down_bn_key in sd:
                    down_conv_w = t2j(down_conv_key)
                    down_bn_w = t2j(f'{block_base}.downsample.1.weight')
                    down_bn_b = t2j(f'{block_base}.downsample.1.bias')
                    down_bn_rm = t2j(f'{block_base}.downsample.1.running_mean')
                    down_bn_rv = t2j(f'{block_base}.downsample.1.running_var')
                    downsample = {
                        'conv_w': down_conv_w,
                        'conv_padding': (0, 0),
                        'bn_w': down_bn_w,
                        'bn_b': down_bn_b,
                        'bn_running_mean': down_bn_rm,
                        'bn_running_var': down_bn_rv,
                        'bn_eps': float(getattr(block_module.downsample[1], 'eps', 1e-5)),
                    }
                else:
                    # Fallback: if state_dict doesn't include downsample keys, skip
                    downsample = None

            block_entry = {
                'params': {
                    'conv1_w': conv1_w,
                    'conv1_padding': conv1_padding,
                    'conv1_dilation': conv1_dilation,
                    'bn1_w': bn1_w,
                    'bn1_b': bn1_b,
                    'bn1_running_mean': bn1_rm,
                    'bn1_running_var': bn1_rv,
                    'bn1_eps': bn1_eps,

                    'conv2_w': conv2_w,
                    'conv2_padding': conv2_padding,
                    'conv2_dilation': conv2_dilation,
                    'bn2_w': bn2_w,
                    'bn2_b': bn2_b,
                    'bn2_running_mean': bn2_rm,
                    'bn2_running_var': bn2_rv,
                    'bn2_eps': bn2_eps,
                },
                'stride': tuple(conv1_stride),
                'downsample': downsample
            }
            layer_list.append(block_entry)

        params[f'layer{li}'] = layer_list

    # final fc - try resnet_prefix+'.fc' first, then fallback to 'fc' at root
    fc_key_w = _make_key(resnet_prefix, 'fc.weight')
    fc_key_b = _make_key(resnet_prefix, 'fc.bias')
    if fc_key_w in sd and fc_key_b in sd:
        params['fc'] = {
            'weight': t2j(fc_key_w),
            'bias': t2j(fc_key_b),
        }
    elif 'fc.weight' in sd and 'fc.bias' in sd:
        params['fc'] = {
            'weight': t2j('fc.weight'),
            'bias': t2j('fc.bias'),
        }
    else:
        raise KeyError("Couldn't find final 'fc' parameters in state_dict under "
                       f"'{fc_key_w}'/'{fc_key_b}' or 'fc.weight'/'fc.bias'")

    return params
# def extract_resnet_params_from_torch(torch_model, resnet_prefix='model'):
#     """
#     torch_model: your full PyTorch model that contains a ResNet submodule.
#       e.g. in your TaxiNetDNN, the ResNet is `model` so default prefix 'model' will work.
#     Returns: params dict that matches the structure expected by resnet18_jax_forward
#     """
#     sd = torch_model.state_dict()  # OrderedDict mapping names -> tensors

#     def t2j(name):
#         # safe getter: convert a torch tensor to jnp.array, else raise KeyError
#         v = sd[name]
#         return jnp.array(v.cpu().numpy())

#     params = {}

#     # top conv + bn + fc names depend on how the PyTorch ResNet was constructed.
#     # Common pattern: prefix + '.conv1.weight', prefix + '.bn1.weight', etc.
#     params['conv1'] = {
#         'weight': t2j(f'{resnet_prefix}.conv1.weight'),
#         'stride': (2, 2),
#         'padding': (3, 3),
#         'dilation': (1, 1),
#     }
#     params['bn1'] = {
#         'weight': t2j(f'{resnet_prefix}.bn1.weight'),
#         'bias': t2j(f'{resnet_prefix}.bn1.bias'),
#         'running_mean': t2j(f'{resnet_prefix}.bn1.running_mean'),
#         'running_var': t2j(f'{resnet_prefix}.bn1.running_var'),
#         'eps': float(getattr(getattr(torch_model, resnet_prefix).bn1, 'eps', 1e-5))
#     }

#     # layers 1..4 each contain blocks, typically named layer1.0.conv1.weight etc.
#     for li in range(1, 5):
#         layer_list = []
#         # find how many blocks in that layer using keys in state_dict
#         # we'll discover block indices by scanning keys
#         prefix_layer = f'{resnet_prefix}.layer{li}.'
#         block_indices = set()
#         for k in sd.keys():
#             if k.startswith(prefix_layer):
#                 rest = k[len(prefix_layer):]
#                 # rest like '0.conv1.weight' or '1.downsample.0.weight'
#                 idx_str = rest.split('.')[0]
#                 try:
#                     idx = int(idx_str)
#                     block_indices.add(idx)
#                 except ValueError:
#                     continue
#         block_indices = sorted(list(block_indices))

#         for idx in block_indices:
#             block_params = {}
#             block_base = f'{resnet_prefix}.layer{li}.{idx}.'
#             # conv1
#             block_params['conv1_w'] = t2j(block_base + 'conv1.weight')
#             # PyTorch BasicBlock convs often have padding (1,1) and dilation (1,1)
#             # we can store them as metadata if needed
#             block_params['conv1_padding'] = (1, 1)
#             block_params['conv1_dilation'] = (1, 1)

#             # bn1
#             block_params['bn1_w'] = t2j(block_base + 'bn1.weight')
#             block_params['bn1_b'] = t2j(block_base + 'bn1.bias')
#             block_params['bn1_running_mean'] = t2j(block_base + 'bn1.running_mean')
#             block_params['bn1_running_var'] = t2j(block_base + 'bn1.running_var')
#             block_params['bn1_eps'] = float(getattr(getattr(getattr(torch_model, resnet_prefix).layer1, '0').bn1, 'eps', 1e-5)) if False else 1e-5
#             # conv2
#             block_params['conv2_w'] = t2j(block_base + 'conv2.weight')
#             block_params['conv2_padding'] = (1, 1)
#             block_params['conv2_dilation'] = (1, 1)
#             # bn2
#             block_params['bn2_w'] = t2j(block_base + 'bn2.weight')
#             block_params['bn2_b'] = t2j(block_base + 'bn2.bias')
#             block_params['bn2_running_mean'] = t2j(block_base + 'bn2.running_mean')
#             block_params['bn2_running_var'] = t2j(block_base + 'bn2.running_var')
#             block_params['bn2_eps'] = 1e-5

#             # stride: if conv1 in this block has stride>1, we'll extract it.
#             # Typical BasicBlock: the conv1 stride is (2,2) only for the first block in layer2/3/4
#             # We can deduce stride by checking if conv1.weight has the same spatial dims but need stride from module
#             # Simpler approach: inspect the module directly if accessible
#             # Try to resolve by walking the torch model modules
#             try:
#                 module_block = dict(getattr(getattr(torch_model, resnet_prefix).layer1, '__dict__'))  # dummy to silence linters
#             except Exception:
#                 module_block = None

#             # Attempt to read stride from the actual module if present
#             try:
#                 block_module = getattr(getattr(torch_model, resnet_prefix), f'layer{li}')[idx]
#                 s = getattr(block_module.conv1, 'stride', (1, 1))
#                 # torch returns tuple like (2,2) or (1,1)
#                 if isinstance(s, tuple):
#                     block_params['stride'] = tuple(s)
#                 else:
#                     block_params['stride'] = (int(s), int(s))
#             except Exception:
#                 block_params['stride'] = (1, 1)

#             # check for downsample submodule
#             downsample_key_base = block_base + 'downsample.'
#             has_downsample = any(k.startswith(downsample_key_base) for k in sd.keys())
#             downsample = None
#             if has_downsample:
#                 # downsample usually: (0): Conv2d, (1): BatchNorm2d
#                 down_conv_w = t2j(block_base + 'downsample.0.weight')
#                 down_bn_w = t2j(block_base + 'downsample.1.weight')
#                 down_bn_b = t2j(block_base + 'downsample.1.bias')
#                 down_bn_rm = t2j(block_base + 'downsample.1.running_mean')
#                 down_bn_rv = t2j(block_base + 'downsample.1.running_var')
#                 # stride of downsample conv is same as conv1 stride
#                 ds = block_params['stride']
#                 downsample = {
#                     'conv_w': down_conv_w,
#                     'conv_padding': (0, 0),
#                     'bn_w': down_bn_w,
#                     'bn_b': down_bn_b,
#                     'bn_running_mean': down_bn_rm,
#                     'bn_running_var': down_bn_rv,
#                     'bn_eps': 1e-5,
#                 }
#             # finalize block wrapper
#             block_entry = {
#                 'params': {
#                     # rename to the keys expected by basic_block_jax
#                     'conv1_w': block_params['conv1_w'],
#                     'conv1_padding': block_params['conv1_padding'],
#                     'conv1_dilation': block_params['conv1_dilation'],
#                     'bn1_w': block_params['bn1_w'],
#                     'bn1_b': block_params['bn1_b'],
#                     'bn1_running_mean': block_params['bn1_running_mean'],
#                     'bn1_running_var': block_params['bn1_running_var'],
#                     'bn1_eps': block_params.get('bn1_eps', 1e-5),

#                     'conv2_w': block_params['conv2_w'],
#                     'conv2_padding': block_params['conv2_padding'],
#                     'conv2_dilation': block_params['conv2_dilation'],
#                     'bn2_w': block_params['bn2_w'],
#                     'bn2_b': block_params['bn2_b'],
#                     'bn2_running_mean': block_params['bn2_running_mean'],
#                     'bn2_running_var': block_params['bn2_running_var'],
#                     'bn2_eps': block_params.get('bn2_eps', 1e-5),
#                 },
#                 'stride': block_params['stride'],
#                 'downsample': downsample
#             }
#             layer_list.append(block_entry)
#         params[f'layer{li}'] = layer_list

#     # final fc - many ResNet definitions store fc under prefix+'.fc'
#     try:
#         params['fc'] = {
#             'weight': t2j(f'{resnet_prefix}.fc.weight'),
#             'bias': t2j(f'{resnet_prefix}.fc.bias'),
#         }
#     except KeyError:
#         # If your top-level model has another fc, try to find it at 'fc' root
#         params['fc'] = {
#             'weight': t2j('fc.weight'),
#             'bias': t2j('fc.bias'),
#         }

#     return params


# -------------------------
# Example usage snippet
# -------------------------
if __name__ == '__main__':
    # ---- Example: load your PyTorch TaxiNetDNN model (replace with your checkpoint) ----
    model = pretrained_dnn.get_network()
    import ipdb; ipdb.set_trace()
    # model.load_state_dict(torch.load('taxinet_checkpoint.pth', map_location='cpu'))
    # model.eval()

    # For demonstration create a torchvision ResNet18 (weights random)
    from torchvision.models import resnet18
    pt_resnet = resnet18(pretrained=False)
    pt_resnet.eval()

    # Extract params
    params = extract_resnet_params_from_torch(pt_resnet, resnet_prefix='')

    # Create random input: NCHW
    import numpy as np
    rng = np.random.RandomState(0)
    x_torch = torch.tensor(rng.randn(2, 3, 224, 224).astype(np.float32))
    x_jax = jnp.array(x_torch.numpy())

    # PyTorch forward (to compare)
    with torch.no_grad():
        out_torch = pt_resnet(x_torch).numpy()

    # JAX forward
    out_jax = resnet18_jax_forward(params, x_jax)
    print("PyTorch output shape:", out_torch.shape)
    print("JAX output shape:", out_jax.shape)
    # You can compare numerically (they may differ if any eps/affine differences exist)
    print("Max abs diff example:", float(np.max(np.abs(out_jax - out_torch))))
