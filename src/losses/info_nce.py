import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    General InfoNCE loss over all 2N samples.

    Given two batches x [B, D] and y [B, D], concatenates them into a
    [2B, D] set of embeddings and computes the contrastive loss where
    each sample's positive is its corresponding pair, and all other
    2B-2 samples are negatives (including within-view negatives).

    Args:
        temperature: scaling factor
        similarity: function that takes (x, y) and returns similarity matrix
    """

    def __init__(self, temperature: float = 0.07, similarity=None):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity if similarity is not None else self._dot_similarity

    @staticmethod
    def _dot_similarity(x, y):
        return torch.matmul(x, y.T)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]  (view 1)
        y: [B, D]  (view 2)
        Returns: scalar loss
        """
        B = x.size(0)
        # Concatenating both views: [2B, D]
        z = torch.cat([x, y], dim=0)

        # Full [2B, 2B] similarity matrix
        sim = self.similarity(z, z) / self.temperature

        # Masking out self-similarity (diagonal)
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, float('-inf'))

        # Labels: positive for sample i is i+B, for sample i+B is i
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss