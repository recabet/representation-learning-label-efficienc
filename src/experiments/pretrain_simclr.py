#!/usr/bin/env python
"""
pretrain_simclr.py

Pretrain SimCLR on unlabeled STL-10 dataset.

Saves checkpoints every 10 epochs.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_handling.datasets import STL10Dataset
from src.models.simclr import SimCLR
from src.training.train import fit
from src.configs.simclr_config import SIMCLR_CONFIG
from src.configs.global_config import GLOBAL_CONFIG

import torch.nn as nn
import torch.optim as optim



simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

class SimCLRDataset(torch.utils.data.Dataset):
    """
    Wrap STL10Dataset to return two views for SimCLR
    """
    def __init__(self, split="unlabeled"):
        self.dataset = STL10Dataset(split=split, transform=simclr_transform, labeled=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        return x, x

train_dataset = SimCLRDataset(split="unlabeled")
train_loader = DataLoader(train_dataset,
                          batch_size=SIMCLR_CONFIG.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)


model = SimCLR(base_model="resnet18", out_dim=128, pretrained=False).to(GLOBAL_CONFIG.DEVICE)


# Use NT-Xent loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        batch_size = z1.size(0)

        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(batch_size*2, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        positives = torch.cat([torch.diag(similarity_matrix, batch_size),
                               torch.diag(similarity_matrix, -batch_size)], dim=0)
        loss = -torch.log(torch.exp(positives) / torch.exp(similarity_matrix).sum(dim=1))
        return loss.mean()

criterion = NTXentLoss(temperature=0.5)
optimizer = optim.Adam(model.parameters(), lr=SIMCLR_CONFIG.LEARNING_RATE)


if __name__ == "__main__":
    fit(model,
        train_loader,
        criterion,
        optimizer,
        GLOBAL_CONFIG.DEVICE,
        epochs=SIMCLR_CONFIG.EPOCHS,
        checkpoint_dir=str(SIMCLR_CONFIG.CHECKPOINT_DIR),)