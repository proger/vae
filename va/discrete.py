import math
from typing import Sequence

import torch
import torch.nn as nn


def make_block(
    in_features: int,
    hidden_features: Sequence[int],
    out_features: int,
    hidden_activation: nn.Module = nn.LeakyReLU,
):
    net = []
    for hidden in hidden_features:
        net.extend([
            nn.Linear(in_features, hidden),
            hidden_activation(),
        ])
        in_features = hidden
    net.extend([
        nn.Linear(in_features, out_features)
    ])
    return nn.Sequential(*net)


def counts_to_frequencies(counts):
    frequencies = counts / counts.sum(dim=1, keepdim=True)
    return frequencies


def log1p_exp(x):
    """
    Computationally stable function for computing log(1+exp(x)).
    """
    x_ = x * x.ge(0).to(torch.float32)
    res = x_ + torch.log1p(torch.exp(-torch.abs(x)))
    return res


def log_prob(x, logits): # KL?
    numerator = - (x * logits).sum(dim=1)
    denominator = - (x * counts_to_frequencies(x).log()).sum(dim=1)
    return numerator - denominator


def entropy(z, z_logits):
    true = z * (-log1p_exp(-z_logits))
    false = (1 - z) * (-log1p_exp(z_logits))
    return true + false + math.log(2)


class VAE(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        latent_features: int = 64,
        decoder_hidden_features: Sequence[int] = (512, )
    ):
        super().__init__()
        
        self.encoder = make_block(
            in_features=vocab_size,
            hidden_features=(),
            out_features=latent_features,
        )
        
        self.decoder = nn.Sequential(
            make_block(
                in_features=latent_features,
                hidden_features=decoder_hidden_features,
                out_features=vocab_size,
            ),
            nn.LogSoftmax(dim=1),
        )
        
    def encode(self, counts):
        z_logits = self.encoder(counts)
        return z_logits

    def elbo(self, z, z_logits, counts):
        logits = self.decoder(z)
        return log_prob(counts, logits) + entropy(z, z_logits.detach()).sum(dim=1)

    def disarm_elbo(self, z_logits, counts):
        """Estimates the ELBO with DisARM estimator. See:

        Dong, Mnih & Tucker (2020): "DisARM: An Antithetic Gradient Estimator for Binary Latent Variables"
        """
        # sample uniform noise
        U = z_logits.new_empty(z_logits.size()).uniform_()

        # ARM expansion
        b1 = (U > torch.sigmoid(-z_logits)).type_as(z_logits)
        b2 = (U < torch.sigmoid(z_logits)).type_as(z_logits)
        f1 = self.elbo(b1, z_logits, counts)
        f2 = self.elbo(b2, z_logits, counts)

        # DisARM generator of grad in a, vector of batch size
        ones, zeros = torch.ones_like(b2), torch.zeros_like(b2)
        inner_term = (-1*ones).pow(b2) * torch.where(b1 != b2, ones, zeros) * z_logits.abs().sigmoid()
        # ARM goes like this:
        # inner_term = 2*U - 1

        # estimates gradient through Bernoulli
        grad_obj = 0.5 * (f1 - f2).detach() * ((inner_term * z_logits).flatten(start_dim=1).sum(dim=1, keepdim=False))
        
        # estimate the loss value and can be differentiated in other parameters
        loss_value = (f1 + f2) / 2
        return loss_value + (grad_obj - grad_obj.detach())

    def make_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
