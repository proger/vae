# [Re] VAE Approximation Error: ELBO and Exponential Families

We successfully reproduce experiments from ICLR2022 paper [VAE Approximation Error: ELBO and Exponential Families](https://openreview.net/forum?id=OIs3SxU5Ynl) by Alexander Shekhovtsov, Dmitrij Schlesinger and Boris Flach.

Below we include our reproduction code.

## Prerequisites

This Python project uses [hatch](https://hatch.pypa.io/latest/intro/) to manage the `va` package.
Training scripts in [./scripts](./scripts) require the package and its dependencies to be installed.

Please install the package using the following command.

```
pip install hatch
pip install -e .
```

To create a virtual environment, you may use `hatch shell`.

## Gaussian Mixtures

This experiment demonstrates a VAE with a factorized encoder with limited representation power.
When trained jointly, the decoder is sacrificing its reconstruction ability to adapt to the encoder.

```
python ./scripts/train_gmm.py
```

This script generates plots as it goes along, check it out!

## Semantic Hashing

This experiment trains a VAE with Bernoulli latent variables on a Semantic Hashing task on the 20 Newsgroups dataset. This model encodes text into a binary code that can be used for search. The best model with DisARM gradient estimator achieves negative ELBO of 387, improving original work.

To reproduce the model using word counts as inputs with the best hyperparameters (including 64 bits latent code), run:

```
python ./scripts/train_semhash.py semhash.pt
```

To train on frequencies as inputs, run:

```
python ./scripts/train_semhash.py --frequencies
```

[scripts/train_semhash_all.sh](scripts/train_semhash_all.sh) reproduces experiments with all hyperparameters in sequence.



## CelebA

Here we pretrain a GAN as a ground truth decoder, pretrain encoder using conditional likelihood and then train a VAE end-to-end, seeing ELBO improve but reconstruction quality gets worse as witnessed by FID.

This experiment is based on [DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

To begin, you need to download **Align&Cropped Images** version (`img_align_celeba.zip`) of the dataset at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and unpack it do `data/celeba`.


```
python ./scripts/train_celeba_01_gan.py  # pretrain a GAN
python ./scripts/train_celeba_02_encoder.py # pretrain encoder
python ./scripts/train_celeba_03_joint.py # train VAE
python ./scripts/train_celeba_04_fid.py # generate images to compute FID
```

To compute FID scores you will need to install [pytorch-fid](https://github.com/mseitzer/pytorch-fid).

```
python -m pytorch_fid exp/dcgan/images exp/dcgan-encoder/images
python -m pytorch_fid exp/dcgan/images exp/dcgan-vae/images
```
