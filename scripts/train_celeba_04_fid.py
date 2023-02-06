from pathlib import Path

import torch.utils.data
import torchvision.utils as vutils

from va.celeba.encoder import Encoder
from va.celeba.generator import Generator


@torch.inference_mode()
def main():
    gt_decoder_path = Path("exp/dcgan/generator-44.pt")
    vae_encoder_dir = Path("exp/dcgan-encoder/")
    vae_fix_dir = Path("exp/dcgan-vae/")

    img_gt_dir = Path("exp/dcgan/images/")
    img_gt_dir.mkdir(parents=True, exist_ok=True)
    img_con_dir = Path("exp/dcgan-encoder/images/")
    img_con_dir.mkdir(parents=True, exist_ok=True)
    img_fix_dir = Path("exp/dcgan-vae/images/")
    img_fix_dir.mkdir(parents=True, exist_ok=True)

    nz = 100
    ngf = 64
    ndf = 64
    nc = 3
    num_epochs = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # ground truth decoder
    netG = Generator(nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(gt_decoder_path))

    # pretrained encoder using conditional likelihood
    encoder_con = Encoder(nz, ndf, nc).to(device)
    encoder_con.load_state_dict(torch.load(vae_encoder_dir / "encoder-990000.pt"))

    # finetuned decoder
    decoder_fix= Generator(nz, ngf, nc).to(device)
    decoder_fix.load_state_dict(torch.load(vae_fix_dir / "decoder-990000.pt"), strict=False)

    # finetuned encoder
    encoder_fix = Encoder(nz, ndf, nc).to(device)
    encoder_fix.load_state_dict(torch.load(vae_fix_dir / "encoder-990000.pt"))

    for epoch in range(num_epochs):
        z_gt = torch.randn(1, nz, 1, 1, device=device)
        xvis_gt = netG(z_gt)
        x_gt_fix = torch.normal(xvis_gt, 0.05)

        z_mu_con, z_sigma_con = encoder_con(x_gt_fix)
        z_mu_fix, z_sigma_fix = encoder_fix(x_gt_fix)

        z_con = torch.normal(z_mu_con, z_sigma_con)
        z_fix = torch.normal(z_mu_fix, z_sigma_fix)
        xvis_con = netG(z_con)
        xvis_fix = torch.normal(decoder_fix(z_fix), 0.05)

        vutils.save_image(xvis_gt, img_gt_dir / f'{epoch}.png', normalize=True, range=(-1,1))
        vutils.save_image(xvis_con, img_con_dir / f'{epoch}.png', normalize=True, range=(-1,1))
        vutils.save_image(xvis_fix, img_fix_dir / f'{epoch}.png', normalize=True, range=(-1,1))

        if epoch % 100 == 0:
            print("generating images: step", epoch, "of", num_epochs)

if __name__ == "__main__":
    main()
