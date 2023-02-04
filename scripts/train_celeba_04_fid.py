from Encoder import Encoder
from EncoderNoSig import EncoderNoSig
from Generator import Generator
from GeneratorNoSig import GeneratorNoSig
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def main():
    ground_path = "check/gan_generator/generator-44.pt"
    vae_encoder_dir = "check/vae_encoder/"
    vae_fix_dir = "check/vae_fix/"
    vae_learn_dir = "check/vae_learn/"

    img_gt_dir = "data/img_gt/"
    img_con_dir = "data/img_encoder/"
    img_fix_dir = "data/img_fix/"
    img_learn_dir = "data/img_learn/"

    nz = 100
    ngf = 64
    ndf = 64
    nc = 3
    num_epoches = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Create Generator
    netG = Generator(nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(ground_path))
    print(netG)
    # Create Encoder of conditional likelihood
    encoder_con = Encoder(nz, ndf, nc).to(device)
    encoder_con.load_state_dict(torch.load(vae_encoder_dir + "encoder-990000.pt"))
    print(encoder_con)

    # Create Decoder of fixed sigma
    decoder_fix = GeneratorNoSig(nz, ngf, nc).to(device)
    decoder_fix.load_state_dict(torch.load(vae_fix_dir + "decoder-990000.pt"))
    print(decoder_fix)
    # Create Encoder of fixed sigma
    encoder_fix = Encoder(nz, ndf, nc).to(device)
    encoder_fix.load_state_dict(torch.load(vae_fix_dir + "encoder-990000.pt"))
    print(encoder_fix)

    # Create Decoder of learned sigma
    decoder_learn = Generator(nz, ngf, nc).to(device)
    decoder_learn.load_state_dict(torch.load(vae_learn_dir + "decoder-990000.pt"))
    print(decoder_learn)
    # Create Encoder of learned sigma
    encoder_learn = Encoder(nz, ndf, nc).to(device)
    encoder_learn.load_state_dict(torch.load(vae_learn_dir + "encoder-990000.pt"))
    print(encoder_learn)

    for epoch in range(num_epoches):
        with torch.no_grad():
            z_gt = torch.randn(1, nz, 1, 1, device=device)
            xvis_gt = netG(z_gt)
            x_gt_fix = torch.normal(xvis_gt, 0.05)
            x_gt_learn = torch.normal(xvis_gt, decoder_learn.sigma)
            # print(x_gt.shape)
        z_mu_con, z_sigma_con = encoder_con(z_gt)
        z_mu_fix, z_sigma_fix = encoder_fix(x_gt_fix)
        z_mu_learn, z_sigma_learn = encoder_learn(x_gt_learn)
        # print(z.shape)
        z_con = torch.normal(z_mu_con, z_sigma_con)
        z_fix = torch.normal(z_mu_fix, z_sigma_fix)
        z_learn = torch.normal(z_mu_learn, z_sigma_learn)
        xvis_con = netG(z_con)
        xvis_fix = torch.normal(decoder_fix(z_fix), 0.05)
        xvis_learn = torch.normal(decoder_learn(z_learn), decoder_learn.sigma)

        vutils.save_image(xvis_gt, img_gt_dir + str(epoch) + '.png', normalize = True, range=(-1,1))
        vutils.save_image(xvis_con, img_con_dir + str(epoch) + '.png', normalize = True, range=(-1,1))
        vutils.save_image(xvis_fix, img_fix_dir + str(epoch) + '.png', normalize = True, range=(-1,1))
        vutils.save_image(xvis_learn, img_learn_dir + str(epoch) + '.png', normalize = True, range=(-1,1))


if __name__ == "__main__":
    main()
