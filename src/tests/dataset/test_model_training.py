import unittest

from src.models.simclr import SimCLR
from src.losses.nt_xent import NTXentLoss
from src.data_handling.datasets import STL10Dataset
from src.configs.global_config import GLOBAL_CONFIG
from src.utils.reproducibility import set_seed, log_environment

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class TestSimCLRTraining(unittest.TestCase):
    def setUp(self):
        set_seed(GLOBAL_CONFIG.SEED)

        log_environment()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        full_dataset = STL10Dataset(split="unlabeled", transform=self.transform, labeled=False)
        self.dataset = Subset(full_dataset, range(min(16, len(full_dataset))))
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

        # Model & loss
        self.model = SimCLR(base_model="resnet18", out_dim=128).to(self.device)
        self.criterion = NTXentLoss(temperature=0.5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)

    def test_forward_pass(self):
        """Test single forward pass"""
        batch = next(iter(self.loader))
        # For unlabeled, returns only images; simulate dual view
        x = batch
        if isinstance(x, (list, tuple)):
            x1 = x2 = x[0].to(self.device)
        else:
            x1 = x2 = x.to(self.device)

        h1, z1 = self.model(x1)
        h2, z2 = self.model(x2)

        self.assertEqual(h1.shape[0], x1.shape[0])
        self.assertEqual(z1.shape[0], x1.shape[0])
        self.assertEqual(z2.shape[1], 128)  # projection dim

    def test_shadow_training_epoch(self):
        """Run one epoch of shadow training"""
        self.model.train()
        total_loss = 0
        for batch in self.loader:
            x = batch
            if isinstance(x, (list, tuple)):
                x1 = x2 = x[0].to(self.device)
            else:
                x1 = x2 = x.to(self.device)

            h1, z1 = self.model(x1)
            h2, z2 = self.model(x2)

            loss = self.criterion(z1, z2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.loader)
        print(f"[Shadow training] Avg loss: {avg_loss:.4f}")
        self.assertGreater(avg_loss, 0)


if __name__ == "__main__":

    unittest.main()
