import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised learning.
    Consists of:
      1. Encoder (ResNet18/50)
      2. Projection head (2-layer MLP)
    """

    def __init__(self,
                 base_model: str = "resnet18",
                 out_dim: int = 128,
                 pretrained: bool = False):

        super(SimCLR, self).__init__()

        # ---- Encoder ----
        if base_model == "resnet18":
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = self.encoder.fc.in_features
        elif base_model == "resnet50":
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = self.encoder.fc.in_features
        else:
            raise ValueError(f"Unknown base_model: {base_model}")

        self.encoder.fc = nn.Identity()

        # ---- Projection head (3-layer MLP, SimCLR v2) ----
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, 96, 96] input images
        Returns:
            h: encoder features
            z: projection head features (for contrastive loss)
        """
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
