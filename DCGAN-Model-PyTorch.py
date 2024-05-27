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
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=1, normalize=True)))
plt.show()

