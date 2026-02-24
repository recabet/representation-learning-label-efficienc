import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-Scaled Cross Entropy Loss for SimCLR
    Reference: https://arxiv.org/abs/2002.05709
    """

    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: [batch_size, feature_dim] - first view
            z2: [batch_size, feature_dim] - second view

        Returns:
            Scalar contrastive loss
        """
        batch_size = z1.size(0)

        # 1. Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 2. Concatenate for 2N batch
        z = torch.cat([z1, z2], dim=0)  # [2*B, D]

        # 3. Cosine similarity matrix
        sim = torch.matmul(z, z.T)  # [2B, 2B]
        sim = sim / self.temperature

        # 4. Remove similarity with self
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)

        # 5. Positive pairs
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels, labels], dim=0)

        # 6. Cross-entropy
        loss = F.cross_entropy(sim, labels)

        return loss