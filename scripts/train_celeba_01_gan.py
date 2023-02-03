"""
Based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb

from va.celeba.Discriminator import Discriminator
from va.celeba.Generator import Generator



def load_data(dataroot, image_size, batch_size, workers):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers
    )

    return dataloader


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    wandb.init(project="ATML", entity="dl_prac")
    data_dir = "/home/zang/Qianbo/ATML/celeba/"
    exp_dir = "/home/zang/Qianbo/ATML/CheckGenerator_sig/"

    # Hyper-parameters
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    image_size = 64
    workers = 16 # Number of workers for dataloader
    batch_size = 128 # Batch size during training
    nc = 3 # Number of channels in the training images. For color images this is 3
    nz = 100 # Size of z latent vector (i.e. size of generator input)
    ngf = 64 # Size of feature maps in generator
    ndf = 64 # Size of feature maps in discriminator
    num_epochs = 100 # Number of training epochs
    lr = 0.0002 # Learning rate for optimizers
    beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    dataloader = load_data(data_dir, image_size, batch_size, workers)

    # Create the Generator
    netG = Generator(nz, ngf, nc).to(device)
    wandb.watch(netG)
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netG.apply(weights_init)
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ndf, nc).to(device)
    wandb.watch(netD)
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    iters = 0
    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            stats = {"lossG": errG.item(), "lossD": errD.item(), "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2}

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                stats["generated"] = wandb.Image(img_list[-1])

            wandb.log(stats)
            iters += 1

        filename = exp_dir + "generator-" + str(epoch) + ".pt"
        torch.save(netG.state_dict(), filename)
        print("saved generator to", filename)


if __name__=="__main__":
    main()