import argparse

import torch
import torch.nn as nn

from va.semhash import SematicHasher
from va.newsgroups import make_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
parser.add_argument('--latent_features', type=int, default=64, help='number of latent Bernoulli variables (bits)')
parser.add_argument('--decoder_hidden_features', type=int, nargs='*', default=(), help='list of hidden layer dimensions for the decoder, one for each layer')
parser.add_argument('output_checkpoint', type=str, help='save the checkpoint')
args = parser.parse_args()

print(vars(args))

torch.manual_seed(1)

model = SematicHasher(
    vocab_size=10000,
    latent_features=args.latent_features,
    decoder_hidden_features=args.decoder_hidden_features
)
model.to(args.device)

print(model)

optimizer = model.make_optimizer(lr=1e-3)
train, test = make_datasets()

print('train documents', len(train))
print('test documents', len(test))

train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)


def train(model, train_loader, optimizer, reinforce_steps=1):
    device = next(model.parameters()).device

    train_loss = 0.
    for word_counts, in train_loader:
        word_counts = word_counts.to(device)
        z_logits = model(word_counts)
        for _ in range(reinforce_steps):
            loss = model.disarm_elbo(z_logits, word_counts).mean() / reinforce_steps
            loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    train_loss /= len(train_loader)
    
    return train_loss

@torch.inference_mode()
def evaluate(model, test_loader):
    device = next(model.parameters()).device

    valid_loss = 0.
    for word_counts, in test_loader:
        word_counts = word_counts.to(device)
        z_logits = model(word_counts)
        loss = model.disarm_elbo(z_logits, word_counts).mean()
        valid_loss += loss.item()
    valid_loss /= len(test_loader)

    return valid_loss

train_losses = []
test_losses = []

for epoch in range(1000):
    train_loss = train(model, train_loader, optimizer)
    test_loss = evaluate(model, test_loader)
    print(f'Epoch {epoch}: train_loss={train_loss:.2f} test_loss={test_loss:.2f}')
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# 500 more epochs with 10 REINFORCE steps
for epoch in range(1000, 1500):
    train_loss = train(model, train_loader, optimizer, reinforce_steps=10)
    test_loss = evaluate(model, test_loader)
    print(f'Epoch {epoch}: train_loss={train_loss:.2f} test_loss={test_loss:.2f}')
    train_losses.append(train_loss)
    test_losses.append(test_loss)

print('Best train loss', min(train_losses))
print('Best test loss', min(test_losses))

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'args': vars(args),
}, args.output_checkpoint)
