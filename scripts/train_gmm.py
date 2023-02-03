from itertools import chain
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from va.gmm import Encoder, Decoder, elbo, kl, log_posterior


mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False

torch.manual_seed(0)

print('🔬 Initializing ground truth GMM and sampling points...')

decoder = Decoder(std=1)

print('Cluster centers:')
print(decoder.theta.weight)

x = decoder.dist().sample((2000,)).reshape(-1, 2)
c = decoder.log_prob(x).argmax(dim=-1)

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=c)
plt.scatter(decoder.theta.weight[0].detach().numpy(), decoder.theta.weight[1].detach().numpy(), color='red', marker='+')
plt.savefig('gmm_ground_truth.pdf')
plt.close(fig)
print('Saved point cloud to gmm_ground_truth.pdf')
print()
print('Pretraining encoder using ELBO...')

encoder = Encoder()
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

log_p_z = torch.tensor([0.25]).log()
with torch.no_grad():
    log_p_xIz = decoder.log_prob(x)
    num, log_p_x = log_posterior(log_p_xIz, log_p_z)

for epoch in range(10000):
    optimizer.zero_grad()
    q_zIx = encoder(x)
    loss = -elbo(q_zIx, log_p_xIz, log_p_z).mean()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        with torch.no_grad():
            true_kl = kl(q_zIx, num - log_p_x).mean().item()
        print(f'epoch={epoch:4d}',
              f'elbo={-loss.item():.4f}',
              f'true_evidence={log_p_x.mean().item():.4f}',
              f'true_kl={true_kl:.4f}')

print(f'epoch={epoch:4d}',
      f'elbo={-loss.item():.4f}',
      f'true_evidence={log_p_x.mean().item():.4f}',
      f'true_kl={true_kl:.4f}')

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=encoder(x).argmax(dim=-1))
plt.scatter(decoder.theta.weight[0].detach().numpy(), decoder.theta.weight[1].detach().numpy(), color='red', marker='+')
plt.savefig('gmm_pretrained_encoder_decision.pdf')
plt.close(fig)
print('Saved gmm_pretrained_encoder_decision.pdf')
print()

print('Joing training of encoder and decoder using ELBO...')

log_p_z = torch.tensor([0.25]).log()
optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=0.0001)

with torch.no_grad():
    log_p_xIz = decoder.log_prob(x)
    true_num, true_log_p_x = log_posterior(log_p_xIz, log_p_z)

for epoch in range(10000):
    optimizer.zero_grad()
    q_zIx = encoder(x)
    log_p_xIz = decoder.log_prob(x)
    loss = -elbo(q_zIx, log_p_xIz, log_p_z).mean()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        with torch.no_grad():
            num, log_p_x = log_posterior(log_p_xIz, log_p_z)
            true_kl = kl(q_zIx, true_num - true_log_p_x).mean().item()
        print(f'epoch={epoch:4d}',
              f'elbo={-loss.item():.4f}',
              f'true_evidence={log_p_x.mean().item():.4f}',
              f'true_kl={true_kl:.4f}')
        
print(f'epoch={epoch:4d}',
      f'elbo={-loss.item():.4f}',
      f'true_evidence={log_p_x.mean().item():.4f}',
      f'true_kl={true_kl:.4f}')

print()
print("Inspecting cluster centers drift:")

print(decoder.theta.weight)

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=encoder(x).argmax(dim=-1))
plt.scatter(decoder.theta.weight[0].detach().numpy(), decoder.theta.weight[1].detach().numpy(), color='red', marker='+')
plt.savefig('gmm_drift.pdf')
plt.close(fig)
print('Saved gmm_drift.pdf 🥳')