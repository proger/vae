import torch
import torch.nn as nn

from va.discrete import VAE
from va.newsgroups import make_datasets

torch.manual_seed(1)

model = VAE(
    vocab_size=10000,
    latent_features=64,
    decoder_hidden_features=()
)
model.to('mps')

print(model)

optimizer = model.make_optimizer(lr=1e-3)
train, test = make_datasets()

train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)


def train(model, train_loader, test_loader, optimizer):
    device = next(model.parameters()).device

    train_loss = 0.
    for word_counts, in train_loader:
        word_counts = word_counts.to(device)
        z_logits = model.encode(word_counts)
        loss = model.disarm_elbo(z_logits, word_counts).mean()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= len(train_loader)

    with torch.inference_mode():
        valid_loss = 0.
        for word_counts, in test_loader:
            word_counts = word_counts.to(device)
            z_logits = model.encode(word_counts)
            loss = model.disarm_elbo(z_logits, word_counts).mean()
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
