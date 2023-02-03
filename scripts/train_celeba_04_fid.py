from Encoder import Encoder
from EncoderNoSig import EncoderNoSig
from Generator import Generator
from GeneratorNoSig import GeneratorNoSig
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def main():
    dir = "/home/zang/Qianbo/ATML/"
    ground_dir = dir + "CheckGenerator_sig/generator-44.pt"

    decoder1_dir = dir + "CheckVAE/decoder-990000.pt"
    encoder1_dir = dir + "CheckVAE/encoder-990000.pt"
    decoder2_dir = dir + "CheckVAE_sig/decoder-990000.pt"
    encoder2_dir = dir + "CheckVAE_sig/encoder-990000.pt"

    exp1_dir = dir + "img1/" 
    exp2_dir = dir + "img2/" 
    exp3_dir = dir + "img3/" 

    nz = 100
    ngf = 64
    ndf = 64
    nc = 3
    num_epoches = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Create Generator
    netG = Generator(nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(ground_dir))
    print(netG)

    # Create Decoder
    decoder = GeneratorNoSig(nz, ngf, nc).to(device)
    decoder.load_state_dict(torch.load(decoder1_dir))
    print(decoder)
    # Create Encoder
    encoder = Encoder(nz, ndf, nc).to(device)
    encoder.load_state_dict(torch.load(encoder1_dir))
    print(encoder)

    # Create Decoder
    decoder_sig = Generator(nz, ngf, nc).to(device)
    decoder_sig.load_state_dict(torch.load(decoder2_dir))
    print(decoder_sig)
    # Create Encoder
    encoder_sig = Encoder(nz, ndf, nc).to(device)
    encoder_sig.load_state_dict(torch.load(encoder2_dir))
    print(encoder_sig)

    for epoch in range(143176, num_epoches):
        with torch.no_grad():
            z_gt = torch.randn(1, nz, 1, 1, device=device)
            xvis_gt = netG(z_gt)

            x1_gt = torch.normal(xvis_gt, 0.05)
            x2_gt = torch.normal(xvis_gt, decoder_sig.sigma)
            # print(x_gt.shape)
        z_mu1, z_sigma1 = encoder(x1_gt)
        z_mu2, z_sigma2 = encoder_sig(x2_gt)
        # print(z.shape)
        z1 = torch.normal(z_mu1, z_sigma1)
        z2 = torch.normal(z_mu2, z_sigma2)

        xvis = torch.normal(decoder(z1), 0.05)
        xvis_sig = torch.normal(decoder(z2), decoder_sig.sigma)

        vutils.save_image(xvis_gt, exp1_dir + str(epoch) + '.png', normalize = True, range=(-1,1))
        vutils.save_image(xvis, exp2_dir + str(epoch) + '.png', normalize = True, range=(-1,1))
        vutils.save_image(xvis_sig, exp3_dir + str(epoch) + '.png', normalize = True, range=(-1,1))


if __name__ == "__main__":
    main()