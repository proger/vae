import torch
import torch.nn as nn
import math


def log_prob_diag(x, mu, sigma=0.01):
    x = x - mu
    p = (x * x / sigma).sum(dim=-1) 
    p = p + math.log(sigma * sigma) + 2 * math.log(2 * math.pi)
    return -0.5 * p


class Decoder(nn.Module):
    "Decoder with hardcoded initialization parameters"

    def __init__(self, std=1):
        super().__init__()
        self.theta = nn.Linear(4, 2, bias=False)
        self.std = std
        self.theta.weight.data = torch.tensor([
            [1,  1, 0, -1],
            [1, -1, 0,  0],
        ], dtype=torch.float32) # (4, 2)

    def log_prob(self, x):
        x = x[:,None,:] # (B, 2) -> (B, 1, 2)
        return log_prob_diag(x, self.theta.weight.T, self.std) # (B, 1, 2) -> (B, 4)

    def dist(self):
        return torch.distributions.MultivariateNormal(
            self.theta.weight.T,
            self.std*torch.eye(2)
        )

class Encoder(nn.Module):
    "Complex encoder that factors over 2-bit latent code"

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.z_b = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32).T

    def forward(self, x):
        g = self.encoder(x).unsqueeze(-1) # (B, 2) -> (B, 2, 1)
        z = g * self.z_b # (B, 2, 1) * (2, 4) -> (B, 2, 4)
        z = z.sum(dim=-2) # (B, 2, 4) -> (B, 4)
        z = z.softmax(dim=-1) # (B, 4)
        return z


def ce(q, log_p):
    return (q * log_p).sum(dim=-1)


def kl(q, log_p):
    return ce(q, q.log() - log_p)


def elbo(q_zIx, log_p_xIz, log_p_z): 
    return ce(q_zIx, log_p_xIz) - kl(q_zIx, log_p_z)


def log_posterior(log_p_xIz, log_p_z):
    num = log_p_xIz + log_p_z
    den = num.logsumexp(dim=-1, keepdim=True)
    return num, den