"""
Joint training of a VAE using ELBO.
"""
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import wandb

from vc.celeba.Encoder import Encoder
from vc.celeba.Generator import Generator

def log_gauss(mu, sigma, x):
    diff = x - mu
    return -torch.log(sigma) - diff*diff/(2*sigma*sigma) - 0.9189


def main():
    wandb.init(project="ATML", entity="dl_prac")
    dir = "/home/zang/Qianbo/ATML/"
    decoder_dir = dir + "CheckGenerator_sig/generator-44.pt"
    encoder_dir = dir + "CheckEncoder_sig/encoder-990000.pt"
    exp_dir = dir + "CheckVAE_sig/"  
    
    nz = 100
    ngf = 64
    ndf = 64
    batch_size = 128
    nc = 3
    lr = 1e-5
    num_epoches = 1000000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epoches,
        "batch_size": batch_size
    }

    # Create Decoder
    decoder_gt = Generator(nz, ngf, nc, learn_sigma=True).to(device)
    decoder_gt.load_state_dict(torch.load(decoder_dir))
    wandb.watch(decoder_gt)
    print(decoder_gt)

    decoder = Generator(nz, ngf, nc, learn_sigma=True).to(device)
    wandb.watch(decoder)
    decoder.load_state_dict(torch.load(decoder_dir))
    print(decoder)

    # Create Encoder
    encoder = Encoder(nz, ndf, nc).to(device)
    wandb.watch(encoder)
    # wandb.watch(encoder)
    encoder.load_state_dict(torch.load(encoder_dir))
    print(encoder)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr) 
    elbo = 0
    z_means = torch.zeros([nz], device=device)
    zz_means = torch.ones([nz], device=device)
    eentr = 94.2 # from log of the previous experiment

    img_list = []
    for epoch in range(num_epoches):
        # generate training batch
        with torch.no_grad():
            z_gt = torch.randn(batch_size, nz, 1, 1, device=device)
            x_gt = torch.normal(decoder_gt(z_gt), decoder.sigma)

        # apply
        z_mu, z_sigma = encoder(x_gt)
        z = torch.normal(torch.zeros_like(z_mu), torch.ones_like(z_sigma))*z_sigma + z_mu
        x_mu = decoder(z)

        optimizer.zero_grad()

        # data term
        logx = log_gauss(x_mu, decoder.sigma, x_gt).sum([1,2,3]).mean() # sumed over [c,h,w], averaged over batch

        # KL term
        kl_div = (z_sigma*z_sigma + z_mu*z_mu - 1 - 2*torch.log(z_sigma)).sum([1,2,3]).mean()/2

        elbo_cur = logx - kl_div
        elbo = elbo*0.999 + elbo_cur.detach()*0.001

        (-elbo_cur).backward()
        optimizer.step()

        # accumulate 
        z_means = z_means*0.999 + z.mean([0,2,3]).detach()*0.001
        zz_means = zz_means*0.999 + (z*z).mean([0,2,3]).detach()*0.001
        eentr = eentr*0.999 + (141.89 + torch.log(z_sigma).detach().sum([1,2,3]).mean())*0.001
        stats = {"elbo": elbo.item(), "z_means": z_means, "zz_means": zz_means, "eentr": eentr}

        if epoch % 10000 == 0:
            file_encoder = exp_dir + "encoder-" + str(epoch) + ".pt"
            file_decoder = exp_dir + "decoder-" + str(epoch) + ".pt"
            torch.save(encoder.state_dict(), file_encoder)
            torch.save(decoder.state_dict(), file_decoder)
            print("saved encoder to", file_encoder)
            print("saved decoder to", file_decoder)

            with torch.no_grad():
                x0 = x_gt[0:8,:,:,:]
                z0 = z_mu[0:8,:,:,:]
                x1 = torch.normal(decoder(z0), decoder.sigma)
                z1 = z[0:8,:,:,:]
                x2 = torch.normal(decoder(z1), decoder.sigma)
                xvis = torch.cat((x0, x1, x2))
                img_list.append(vutils.make_grid(xvis, padding=2, normalize=True))
                stats["generated"] = wandb.Image(img_list[-1]) 
        wandb.log(stats)   

if __name__ == "__main__":
    main()