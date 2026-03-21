import torch.nn as nn

class LinearProbe(nn.Module):
    """
    Linear classifier for evaluating pretrained encoder features.
    Input: features from encoder (h)
    Output: class logits
    """

    def __init__(self, feat_dim: int, num_classes: int = 10):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)