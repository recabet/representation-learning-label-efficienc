from src.losses.info_nce import InfoNCE

import torch
import torch.nn.functional as F

class NTXentLoss(InfoNCE):
    """
    NT-Xent loss (SimCLR version of InfoNCE)

    - Cosine similarity
    - L2 normalization
    - Symmetric loss (2 directions)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__(temperature=temperature)

    @staticmethod
    def _cosine_similarity(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return torch.matmul(x, y.T)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        self.similarity = self._cosine_similarity

        # Symmetric InfoNCE
        loss_12 = super().forward(z1, z2)
        loss_21 = super().forward(z2, z1)

        return (loss_12 + loss_21) / 2