import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    General InfoNCE loss.

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

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        query: [B, D]
        key:   [B, D]
        """

        logits = self.similarity(query, key)
        logits = logits / self.temperature

        labels = torch.arange(query.size(0), device=query.device)

        loss = F.cross_entropy(logits, labels)
        return loss