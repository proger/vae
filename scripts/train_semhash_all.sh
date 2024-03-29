#!/usr/bin/env bash

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
