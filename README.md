# [Re] VAE Approximation Error: ELBO and Exponential Families

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

This experiment demonstrates a VAE with a factorized encoder with not limited representation power.
When trained jointly, the decoder is sacrificing its reconstruction ability to adapt to the encoder.

```
python ./scripts/train_gmm.py
```

## Semantic Hashing

This experiment trains a Bernoulli VAE on a Semantic Hashing task on the 20 Newsgroups dataset.
To reproduce the model with the best hyperparameters, run:

```
python ./scripts/train_semhash.py semhash.pt
```


The following list of commands reproduces all experiments:

```
python ./scripts/train_semhash.py --decoder_hidden_features 512 --latent_features 16 semhash_e1_bits16_dhidden1.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 --latent_features 32 semhash_e1_bits32_dhidden1.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 --latent_features 64 semhash_e1_bits64_dhidden1.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 --latent_features 8 semhash_e1_bits8_dhidden1.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 512 --latent_features 16 semhash_e1_bits16_dhidden2.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 512 --latent_features 32 semhash_e1_bits32_dhidden2.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 512 --latent_features 64 semhash_e1_bits64_dhidden2.pt
python ./scripts/train_semhash.py --decoder_hidden_features 512 512 --latent_features 8 semhash_e1_bits8_dhidden2.pt
python ./scripts/train_semhash.py --latent_features 16 semhash_e1_bits16_dhidden0.pt
python ./scripts/train_semhash.py --latent_features 32 semhash_e1_bits32_dhidden0.pt
python ./scripts/train_semhash.py --latent_features 64 semhash_e1_bits64_dhidden0.pt
python ./scripts/train_semhash.py --latent_features 8 semhash_e1_bits8_dhidden0.pt

python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 --latent_features 16 semhash_e2_bits16_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 --latent_features 32 semhash_e2_bits32_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 --latent_features 64 semhash_e2_bits64_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 --latent_features 8 semhash_e2_bits8_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 512 --latent_features 16 semhash_e2_bits16_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 512 --latent_features 32 semhash_e2_bits32_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 512 --latent_features 64 semhash_e2_bits64_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --decoder_hidden_features 512 512 --latent_features 8 semhash_e2_bits8_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --latent_features 16 semhash_e2_bits16_dhidden0.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --latent_features 32 semhash_e2_bits32_dhidden0.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --latent_features 64 semhash_e2_bits64_dhidden0.pt
python ./scripts/train_semhash.py --frequencies --encoder_hidden_features 512 --latent_features 8 semhash_e2_bits8_dhidden0.pt

python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 --latent_features 16 semhash_e3_bits16_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 --latent_features 32 semhash_e3_bits32_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 --latent_features 64 semhash_e3_bits64_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 --latent_features 8 semhash_e3_bits8_dhidden1.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 512 --latent_features 16 semhash_e3_bits16_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 512 --latent_features 32 semhash_e3_bits32_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 512 --latent_features 64 semhash_e3_bits64_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --decoder_hidden_features 512 512 --latent_features 8 semhash_e3_bits8_dhidden2.pt
python ./scripts/train_semhash.py --frequencies --latent_features 16 semhash_e3_bits16_dhidden0.pt
python ./scripts/train_semhash.py --frequencies --latent_features 32 semhash_e3_bits32_dhidden0.pt
python ./scripts/train_semhash.py --frequencies --latent_features 64 semhash_e3_bits64_dhidden0.pt
python ./scripts/train_semhash.py --frequencies --latent_features 8 semhash_e3_bits8_dhidden0.pt
```


## CelebA

This experiment is based on [DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

To begin, you need to download **Align&Cropped Images** version (`img_align_celeba.zip`) of the dataset at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and unpack it do `data/`.

```
python ./scripts/train_celeba_01_gan.py
python ./scripts/train_celeba_02_encoder.py
python ./scripts/train_celeba_03_encoder.py
python ./scripts/train_celeba_04_fid.py
```
