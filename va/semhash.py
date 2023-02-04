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


def log_prob(x, logits):
    numerator = - (x * logits).sum(dim=1)
    denominator = - (x * counts_to_frequencies(x).log()).sum(dim=1)
    return numerator - denominator


def entropy(code, code_logits):
    true = code * (-log1p_exp(-code_logits))
    false = (1 - code) * (-log1p_exp(code_logits))
    return (true + false + math.log(2)).sum(dim=1)


class SematicHasher(nn.Module):
    "Bernoulli VAE for Semantic Hashing"
    def __init__(
        self,
        vocab_size: int = 10000,
        latent_features: int = 64,
        encoder_hidden_features: Sequence[int] = (),
        decoder_hidden_features: Sequence[int] = (512, )
    ):
        super().__init__()
        
        self.encoder = make_block(
            in_features=vocab_size,
            hidden_features=encoder_hidden_features,
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
        
    def forward(self, word_counts):
        "Forward the encoder: returns document hash code logits"
        code_logits = self.encoder(word_counts)
        return code_logits
    
    def log_prob(self, word_counts, code):
        "Log probability of the input word counts under the decoder aka reconstruction loss"
        word_logits = self.decoder(code)
        return log_prob(word_counts, word_logits)

    def elbo(self, code, code_logits, word_counts):
        "ELBO with stochastic entropy (based on sampled binary latent code)"
        log_prob = self.log_prob(word_counts, code)
        return log_prob + entropy(code, code_logits.detach())

    def sample_elbo(self, code_logits, word_counts):
        "ELBO with analytic entropy (based on sigmoid of code_logits)"
        uniform = torch.rand_like(code_logits)
        right = (uniform < torch.sigmoid( code_logits)).type_as(code_logits)

        log_prob = self.log_prob(word_counts, right)
        return log_prob + entropy(code_logits.sigmoid(), code_logits)

    def disarm_elbo(self, code_logits, word_counts):
        """ELBO with gradient estimates using DisARM. See:

        Dong, Mnih & Tucker (2020): "DisARM: An Antithetic Gradient Estimator for Binary Latent Variables"
        """
        uniform = torch.rand_like(code_logits)

        # antithetic augmentation
        left  = (uniform > torch.sigmoid(-code_logits)).type_as(code_logits)
        right = (uniform < torch.sigmoid( code_logits)).type_as(code_logits)
        left_loss  = self.elbo(left,  code_logits, word_counts)
        right_loss = self.elbo(right, code_logits, word_counts)

        # DisARM variance reduction
        ones, zeros = torch.ones_like(right), torch.zeros_like(right)
        inner_term = torch.pow(-1*ones, right) * torch.where(left != right, ones, zeros) * torch.sigmoid(torch.abs(code_logits))
        ## ARM does this instead:
        # inner_term = 2*uniform - 1

        # gradient estimate
        grad_obj = 0.5 * (left_loss - right_loss).detach() * ((inner_term * code_logits).flatten(start_dim=1).sum(dim=1, keepdim=False))
        
        # loss estimate
        avg_loss = 0.5 * (left_loss + right_loss)
        return avg_loss + (grad_obj - grad_obj.detach())

    def make_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
