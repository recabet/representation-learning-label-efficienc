#!/usr/bin/env python
"""
model_analysis.py

Analyzes a PyTorch model:
- Prints architecture
- Counts total/trainable/non-trainable parameters
- Plots model graph (requires torchviz)
"""

import torch
import torch.nn as nn
from torchsummary import summary
from pathlib import Path


try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False


def analyze_model(model: nn.Module,
                  input_shape=(3, 96, 96), 
                  save_dir: Path = None):
    """
    Prints model summary, parameter counts, and optionally saves model graph.
    Args:
        model: PyTorch model
        input_shape: Input shape tuple (C,H,W)
        save_dir: Path to save model graph (optional)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("\n========== MODEL ARCHITECTURE ==========")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n========== PARAMETER COUNTS ==========")
    print(f"Total params       : {total_params:,}")
    print(f"Trainable params   : {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

    print("\n========== LAYER-WISE SUMMARY ==========")
    try:
        summary(model, input_size=input_shape)
    except:
        print("Could not run torchsummary. Skipping layer-wise summary.")
    print(TORCHVIZ_AVAILABLE)
    if TORCHVIZ_AVAILABLE and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.randn(1, *input_shape).to(device)
        dot = make_dot(model(dummy_input)[0] if isinstance(model(dummy_input), tuple) else model(dummy_input),
                       params=dict(model.named_parameters()))
        dot.format = "png"
        graph_path = save_dir / "model_graph.png"
        dot.render(str(graph_path), cleanup=True)
        print(f"\nModel graph saved to: {graph_path}")


# ===== Example usage =====
if __name__ == "__main__":
    from src.models.simclr import SimCLR
    from src.models.linear_probe import LinearProbe

    save_dir = Path("./model_analysis")

    simclr = SimCLR(base_model="resnet18", out_dim=128, pretrained=False)
    analyze_model(simclr, input_shape=(3, 96, 96), save_dir=save_dir)

    feat_dim = 512
    linear_probe = LinearProbe(feat_dim=feat_dim, num_classes=10)
    analyze_model(linear_probe, input_shape=(feat_dim,), save_dir=save_dir)