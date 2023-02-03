import torch
import torch.nn as nn

from va.discrete import VAE
from va.newsgroups import create_data

torch.manual_seed(1)

model = VAE(
    vocab_size=10000,
    latent_features=64,
    decoder_hidden_features=()
)
model.to('mps')

print(model)

optimizer = model.make_optimizer(lr=1e-3)
train, train_loader, _, test, test_loader = create_data(batch_size=256)

print(train[0][0].shape)

def train(model, train_loader, test_loader, optimizer):
    device = next(model.parameters()).device

    train_loss = 0.
    for i, (counts, categories) in enumerate(train_loader):
        counts = counts.to(device)
        z_logits = model.encode(counts)
        loss = model.disarm_elbo(z_logits, counts).mean()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= len(train_loader)

    with torch.inference_mode():
        valid_loss = 0.
        for i, (counts, categories) in enumerate(test_loader):
            counts = counts.to(device)
            z_logits = model.encode(counts)
            loss = model.disarm_elbo(z_logits, counts).mean()
            valid_loss += loss.item()
        valid_loss /= len(test_loader)
    return train_loss, valid_loss

for epoch in range(1500):
    train_loss, test_loss = train(model, train_loader, test_loader, optimizer)
    print(f'Epoch {epoch}: train_loss={train_loss:.2f} test_loss={test_loss:.2f}')

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}, 'model.pt')
