import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import zipfile
import matplotlib.animation as animation
from IPython.display import HTML

batch_size = 128
z_dim = 100  # 100-dimensional noise vector as the input to generator
num_epoch = 3
learning_rate = 0.0002  # Optimizer learning rate
betal = 0.5  # Momentum value for Adam optimizer

# with zipfile.ZipFile('img_align_celeba.zip') as zip_ref:
#     zip_ref.extractall('./data/celeba')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
root = './data/celeba'
dataset = datasets.ImageFolder(root=root,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

real_batch = next(iter(dataloader))
plt.figure(figsize=(7, 7))
plt.axis("off")
plt.title("Training Images")
grid = vutils.make_grid(real_batch[0].to('cpu')[:64], padding=1, normalize=True)
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()


# Part 30 - PyTorch Course Training

# Model Implementation

# Weight Initialization

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)


# Discriminator Model

discriminator = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False),
    nn.Sigmoid()
)

disc = discriminator.to(device)
disc.apply(init_weights)

generator = nn.Sequential(

    nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

gen = generator.to(device)
gen.apply(init_weights)

# Training

# Criterion and Optimizers

criterion = nn.BCELoss()
disc_optim = optim.Adam(disc.parameters(), lr=learning_rate, betas=(betal, 0.999))
gen_optim = optim.Adam(gen.parameters(), lr=learning_rate, betas=(betal, 0.999))

fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
real_label = 1.
fake_label = 0.

# Part 31 - Pytorch training course

# Training Loop

D_losses = []
G_losses = []
img_list = []
iters = 0

for epoch in range(num_epoch):
    for i, batch in enumerate(dataloader):
        disc.zero_grad()
        real_images = batch[0].to(device)
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output = disc(real_images).view(-1)  # Pass train images (real) to the disc
        disc_error_real = criterion(output, label)
        disc_error_real.backward()

        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = gen(noise)
        label.fill_(fake_label)
        output = disc(fake.detach()).view(-1)  # Pass fake images to disc
        disc_error_fake = criterion(output, label)
        disc_error_fake.backward()  # Backprob
        disc_error = disc_error_fake + disc_error_real
        disc_optim.step()  # Update disc

        gen.zero_grad()
        label.fill_(real_label)
        output = disc(fake).view(-1)  # Pass fake images to update list
        gen_error = criterion(output, label)  # Gen loss based on the cases in which disc is wrong.
        gen_error.backward()  # Backprob
        gen_optim.step()  # Update gen

        if i % 200 == 0:
            print(
                f"Epoch{[epoch + 1]} | Batch[{i} / {len(dataloader)}] | Disc-Loss: {disc_error.item():.4f} | Gen-Loss: {gen_error.item():.4f}")

        G_losses.append(gen_error.item())
        D_losses.append(disc_error.item())

        if (iters % 500 == 0) or ((epoch == num_epoch - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Gen and Disc Training Loss")
plt.plot(G_losses, label="Gen")
plt.plot(D_losses, label="Disc")
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
imgs = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, imgs, interval=1000, repeat_delay=1000, blit=True)


ani.save("animation_output.html", writer="html")

