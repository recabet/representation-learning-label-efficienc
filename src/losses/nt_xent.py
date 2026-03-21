from src.losses.info_nce import InfoNCE

import torch
import torch.nn.functional as F

class NTXentLoss(InfoNCE):
    """
    NT-Xent loss (SimCLR version of InfoNCE)

    - Cosine similarity
    - L2 normalization
    - Full 2N-sample formulation (symmetric by construction)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__(temperature=temperature)

    @staticmethod
    def _cosine_similarity(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return torch.matmul(x, y.T)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        self.similarity = self._cosine_similarity
        return super().forward(z1, z2)
