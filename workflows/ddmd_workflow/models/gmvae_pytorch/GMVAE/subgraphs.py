import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

use_batch_norm = False

def gaussian_sample(mean, var):
    """
    Gaussian reparameterization trick: z = mean + std * epsilon
    Allows backpropagation through stochastic sampling.
    """
    std = torch.sqrt(torch.clamp(var, min=1e-7))
    eps = torch.randn_like(std)
    return mean + eps * std

def log_normal(x, mu, var, eps=1e-5, dim=-1):
    """Compute log probability of x under Gaussian N(mu, var)"""
    x = x.to(torch.float32)
    mu = mu.to(torch.float32)
    var = var.to(torch.float32)

    if eps > 0.0:
        var = var + eps

    log_2pi = torch.tensor(np.log(2.0 * np.pi), dtype=torch.float32, device=x.device)

    log_prob = -0.5 * torch.sum(
        log_2pi + torch.log(var) + ((x - mu) ** 2) / var,
        dim=dim
    )

    return log_prob

#===============================================================
# Layer builders
#===============================================================

def build_qy_layers(k, hidden_dims=[16, 16]):
    """Build layers for q(y|x) network - cluster assignment"""
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_dim, momentum=0.01, eps=1e-3))

    hidden_stack = nn.Sequential(*layers)
    output_layer = nn.LazyLinear(k)

    return hidden_stack, output_layer

def build_qz_layers(n_z, k, hidden_dims=[16, 16]):
    """Build layers for q(z|x,y) network - inference network for latent variable"""
    y_transform_layer = nn.Sequential(
        nn.LazyLinear(k),
        nn.ReLU()
    )

    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_dim, momentum=0.01, eps=1e-3))

    hidden_stack = nn.Sequential(*layers)

    # Linear mean output, Softplus for strictly positive variance
    zm_layer = nn.LazyLinear(n_z)
    zv_layer = nn.Sequential(
        nn.LazyLinear(n_z),
        nn.Softplus()
    )

    return y_transform_layer, hidden_stack, zm_layer, zv_layer

def build_pz_layers(n_z, hidden_dims=[16]):
    """Build layers for p(z|y) network - prior network for latent variable"""
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_dim, momentum=0.01, eps=1e-3))

    hidden_stack = nn.Sequential(*layers)

    zm_layer = nn.LazyLinear(n_z)
    zv_layer = nn.Sequential(
        nn.LazyLinear(n_z),
        nn.Softplus()
    )

    return hidden_stack, zm_layer, zv_layer

def build_px_layers(n_x, hidden_dims=[16, 16]):
    """Build layers for p(x|z) network - generative network for reconstruction"""
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_dim, momentum=0.01, eps=1e-3))

    hidden_stack = nn.Sequential(*layers)

    xm_layer = nn.LazyLinear(n_x)
    xv_layer = nn.Sequential(
        nn.LazyLinear(n_x),
        nn.Softplus()
    )

    return hidden_stack, xm_layer, xv_layer

#===============================================================
# Execution functions
#===============================================================

def execute_qy_graphs(x, layers, output_layer, training=False):
    """Execute q(y|x) network using pre-built layers"""
    h = layers(x)
    qy_logit = output_layer(h)
    qy = F.softmax(qy_logit, dim=-1)
    return qy_logit, qy

def execute_qz_graph(x, y, y_transform_layer, hidden_layers, zm_layer, zv_layer, training=False):
    """Execute q(z|x,y) network using pre-built layers"""
    h0 = y_transform_layer(y)
    xy = torch.cat([x, h0], dim=1)
    h = hidden_layers(xy)

    zm = zm_layer(h)
    zv = zv_layer(h) + 1e-5

    z = gaussian_sample(zm, zv)
    return z, zm, zv

def execute_pz_graph(y, hidden_layers, zm_layer, zv_layer, training=False):
    """Execute p(z|y) prior network using pre-built layers"""
    h = hidden_layers(y)

    zm = zm_layer(h)
    zv = zv_layer(h) + 1e-5

    return y, zm, zv

def execute_px_graph(z, hidden_layers, xm_layer, xv_layer, training=False):
    """Execute p(x|z) generative/decoder network using pre-built layers"""
    h = hidden_layers(z)

    xm = xm_layer(h)
    xv = xv_layer(h) + 1e-5

    return xm, xv

def labeled_loss(k, x, xm, xv, z, zm, zv, zm_prior, zv_prior, likelihood="gaussian"):
    """
    Variational loss for GMVAE mixture component
    Loss = -log p(x|z) + KL(q(z|x,y) || p(z|y)) - log(1/k)

    likelihood="gaussian": p(x|z) = N(xm, xv), for continuous inputs.
    likelihood="bernoulli": p(x|z) = Bernoulli(sigmoid(xm)), for binary
    inputs such as contact maps. xm are logits and xv is unused.
    """
    if likelihood == "bernoulli":
        reconstruction_loss = F.binary_cross_entropy_with_logits(
            xm, x.to(torch.float32), reduction="none"
        ).sum(dim=-1)
    elif likelihood == "gaussian":
        reconstruction_loss = -log_normal(x, xm, xv)
    else:
        raise ValueError(f"Invalid likelihood: {likelihood}")
    kl_loss = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    mixture_prior = -torch.tensor(np.log(1.0 / k), dtype=torch.float32, device=x.device)

    total_loss = reconstruction_loss + kl_loss + mixture_prior
    return total_loss
