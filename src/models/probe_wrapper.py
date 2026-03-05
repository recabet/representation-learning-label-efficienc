import torch.nn as nn


class ProbeWrapper(nn.Module):
    def __init__(self, encoder, probe):
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, x):
        features = self.encoder(x)
        return self.probe(features)
