import wandb

import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from va.celeba.encoder import Encoder
from va.celeba.generator import Generator


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    wandb.init()
    generator_path = "exp/dcgan/generator-44.pt"
    encoder_check_dir = "exp/dcgan-encoder/"

    nz = 100
    ngf = 64
    ndf = 64
    batch_size = 128
    nc = 3
    lr = 0.00002 
    num_epoches = 1000000
    beta1 = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epoches,
        "batch_size": batch_size
    }

    # Create Decoder
    decoder = Generator(nz, ngf, nc, learn_sigma=True).to(device)
    wandb.watch(decoder)
    decoder.load_state_dict(torch.load(generator_path))
    print(decoder)
    # Create Encoder
    encoder = Encoder(nz, ndf, nc).to(device)
    wandb.watch(encoder)
    # wandb.watch(encoder)
    encoder.apply(weights_init)
    print(encoder)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))

    z_means = torch.zeros([nz], device=device)
    zz_means = torch.zeros([nz], device=device)
    acc_loss = 141.89 # 0.5*ln(2*pi*e)*100
    eentr = 141.89

    img_list = []
    for epoch in range(num_epoches):
        with torch.no_grad():
            z_gt = torch.randn(batch_size, nz, 1, 1, device=device)
            x_gt = torch.normal(decoder(z_gt), decoder.sigma)
        encoder.zero_grad()
        z_mu, z_sigma = encoder(x_gt)
        diff = z_mu - z_gt
        loss = torch.log(z_sigma) + (diff*diff)/(2*z_sigma*z_sigma) + 0.9189
        loss = loss.sum(1).mean()
        loss.backward()
        optimizer.step()

        # accumulate induced prior
        z_sampled = torch.normal(z_mu,z_sigma).detach()
        z_means = z_means*0.999 + z_sampled.mean([0,2,3])*0.001
        zz_means = zz_means*0.999 + (z_sampled*z_sampled).mean([0,2,3])*0.001
        acc_loss = acc_loss*0.999 + loss.detach()*0.001
        eentr = eentr*0.999 + (141.89 + torch.log(z_sigma).detach().sum(1).mean())*0.001
        stats = {"loss": loss.item(), "z_means": z_means, "zz_means": zz_means, "acc_loss": acc_loss, "eentr": eentr}

        if epoch % 10000 == 0:
            filename = encoder_check_dir + "encoder-" +str(epoch) + ".pt"
            torch.save(encoder.state_dict(), filename)
            print("saved encoder to", filename)

            with torch.no_grad():
                x0 = x_gt[0:8,:,:,:]
                z0 = z_mu[0:8,:,:,:]
                x1 = torch.normal(decoder(z0), decoder.sigma)
                z1 = torch.normal(z0, z_sigma[0:8,:,:,:])
                x2 = torch.normal(decoder(z1), decoder.sigma)
                xvis = torch.cat((x0,x1,x2))
                img_list.append(vutils.make_grid(xvis, padding=2, normalize=True))
                stats["generated"] = wandb.Image(img_list[-1]) 
        wandb.log(stats)
        

if __name__ == "__main__":
    main()
