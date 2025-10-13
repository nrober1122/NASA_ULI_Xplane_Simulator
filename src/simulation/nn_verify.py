import os
import time

import cv2
import mss
import numpy as np
import torch
import jax
jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp
import jax_verify
from jax_verify.src import bound_propagation


from train_DNN.model_taxinet import TaxiNetDNN
from utils.torch2jax import torch2jax
from tiny_taxinet2 import dynamics, dynamics_sin

num_samples = 5000

key = jax.random.PRNGKey(int(time.time()))
min_x = jnp.array([-1.0, -45.0, -1.0])
max_x = jnp.array([1.0, 45.0, 1.0])
obs = jax.random.uniform(key, shape=(num_samples, 3), minval=min_x, maxval=max_x)
# obs = jax.random.uniform(key, shape=(num_samples, 3), minval=-1, maxval=1)


def dynamics_wrapper(x):
    cte = x[0]
    he = x[1]
    u = x[2]
    x_next = dynamics(
        cte,
        0.0,
        he,
        u,
        dt=0.1,
    )
    return x_next  # return cte and he


def dynamics_sin_wrapper(x):
    cte = x[0]
    he = x[1]
    u = x[2]
    x_next = dynamics_sin(
        cte,
        0.0,
        he,
        u,  # use sin for steering angle
        dt=0.1,
    )
    return x_next  # return cte and he


x_next_arr = jax.vmap(dynamics_wrapper)(obs)
x_next_arr_sin = jax.vmap(dynamics_sin_wrapper)(obs)
print("Is there any difference?:", jnp.max(jnp.abs(x_next_arr - x_next_arr_sin)))

interval_bound = jax_verify.IntervalBound(
    min_x,
    max_x
)

bounds = jax_verify.backward_crown_bound_propagation(dynamics_wrapper, interval_bound)

import matplotlib.pyplot as plt

# Rectangle bounds
rect_lower = bounds.lower
rect_upper = bounds.upper

# Plot rectangle
plt.figure(figsize=(6, 6))
plt.gca().add_patch(
    plt.Rectangle(
        (rect_lower[0], rect_lower[1]),
        rect_upper[0] - rect_lower[0],
        rect_upper[1] - rect_lower[1],
        fill=False,
        edgecolor='red',
        linewidth=2,
        label='Bounds'
    )
)

# Scatter samples
plt.scatter(x_next_arr[:, 0], x_next_arr[:, 1], s=10, alpha=0.5, label='Samples', color='green')
plt.scatter(x_next_arr_sin[:, 0], x_next_arr_sin[:, 1], s=10, alpha=0.5, label='Samples (sin)', color='blue')

plt.xlabel('cte_next')
plt.ylabel('he_next')
plt.legend()
plt.title('Dynamics Next State Bounds and Samples')
plt.grid(True)
plt.show()


# for i in range(num_samples):
#     x = obs[i]
#     x_next = dynamics_wrapper(x)
#     print("x:", x)
#     print("x_next:", x_next)

# import ipdb; ipdb.set_trace()