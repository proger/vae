import torch
import torch.nn as nn


class EncoderNoSig(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(EncoderNoSig, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.last_layer = nn.Conv2d(ndf * 8, 2*nz, 4, 1, 0, bias=True)
        self.last_layer.weight.data.zero_()
        self.last_layer.bias.data.zero_()

    def forward(self, z):
        scores = self.main(z)
        scores = self.last_layer(scores)
        mu, lsigma = torch.split(scores, self.nz, dim=1)
        return mu, torch.exp(lsigma)