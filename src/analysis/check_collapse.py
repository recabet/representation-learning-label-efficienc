"""
Diagnostic script to check for representation collapse.
Loads the latest SimCLR checkpoint and measures:
  - Standard deviation of features (per dimension and overall)
  - Average pairwise cosine similarity
  - Per-sample norm statistics

If std ≈ 0 or avg pairwise similarity ≈ 1, representations have collapsed.
"""

import sys
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simclr import SimCLR
from src.data_handling.datasets import STL10Dataset
from src.configs.simclr_config import SIMCLR_CONFIG
from src.configs.global_config import GLOBAL_CONFIG


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the checkpoint with the highest epoch number."""
    pattern = re.compile(r"simclr_epoch_(\d+)\.pth$")
    best_epoch, best_path = -1, None
    for p in checkpoint_dir.glob("simclr_epoch_*.pth"):
        m = pattern.search(p.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best_path = epoch, p
    if best_path is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return best_path, best_epoch


@torch.no_grad()
def analyse(checkpoint_path: Path, device: str, num_batches: int = 10):
    # ── load model ───────────────────────────────────────────────────────────
    cfg = SIMCLR_CONFIG()
    model = SimCLR(
        base_model=cfg.BASE_MODEL,
        out_dim=cfg.OUT_DIM,
        pretrained=False,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"✔ Loaded checkpoint: {checkpoint_path}\n")

    # ── dataloader (plain transform, no augmentation) ────────────────────────
    eval_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = STL10Dataset(split="test", transform=eval_transform, labeled=False)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=0, drop_last=False)

    all_h, all_z = [], []
    for i, x in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device)
        h, z = model(x)
        all_h.append(h)
        all_z.append(z)

    all_h = torch.cat(all_h, dim=0)  # encoder output
    all_z = torch.cat(all_z, dim=0)  # projection head output

    print(f"Samples analysed: {all_h.shape[0]}")
    print(f"Encoder dim (h):  {all_h.shape[1]}")
    print(f"Projector dim (z): {all_z.shape[1]}")
    print("=" * 60)

    for name, feats in [("Encoder (h)", all_h), ("Projection head (z)", all_z)]:
        print(f"\n── {name} ──")

        # L2-normalise for cosine analysis
        feats_norm = F.normalize(feats, dim=1)

        # 1. Feature-wise std  (should be ~0.03-0.1 for healthy training)
        std_per_dim = feats.std(dim=0)  # [feat_dim]
        print(f"  Std per dimension  — mean: {std_per_dim.mean():.6f}  "
              f"min: {std_per_dim.min():.6f}  max: {std_per_dim.max():.6f}")

        # 2. Overall std across ALL elements
        overall_std = feats.std().item()
        print(f"  Overall std:         {overall_std:.6f}")

        # 3. Average pairwise cosine similarity (excluding diagonal)
        sim_matrix = feats_norm @ feats_norm.T
        n = sim_matrix.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        avg_cos_sim = sim_matrix[mask].mean().item()
        std_cos_sim = sim_matrix[mask].std().item()
        print(f"  Avg pairwise cosine sim: {avg_cos_sim:.6f}  (std: {std_cos_sim:.6f})")

        # 4. Norms of feature vectors
        norms = feats.norm(dim=1)
        print(f"  Norm — mean: {norms.mean():.4f}  std: {norms.std():.4f}  "
              f"min: {norms.min():.4f}  max: {norms.max():.4f}")

        # 5. Dead dimensions (std ≈ 0)
        dead_dims = (std_per_dim < 1e-5).sum().item()
        print(f"  Dead dimensions (std < 1e-5): {dead_dims}/{feats.shape[1]}")

    # ── verdict ──────────────────────────────────────────────────────────────
    z_norm = F.normalize(all_z, dim=1)
    sim = z_norm @ z_norm.T
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    avg_sim = sim[mask].mean().item()
    z_std = all_z.std(dim=0).mean().item()

    print("\n" + "=" * 60)
    print("VERDICT:")
    if avg_sim > 0.95:
        print("  ⚠️  COLLAPSE DETECTED — avg cosine similarity > 0.95")
        print("     All representations are nearly identical.")
    elif avg_sim > 0.8:
        print("  ⚠️  PARTIAL COLLAPSE — avg cosine similarity > 0.8")
        print("     Representations lack diversity.")
    elif z_std < 0.01:
        print("  ⚠️  LOW VARIANCE — feature std very small")
        print("     Model may be heading toward collapse.")
    else:
        print("  ✅  No obvious collapse detected.")
        print(f"     avg_cosine_sim={avg_sim:.4f}, feature_std={z_std:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    device = GLOBAL_CONFIG.DEVICE
    print(f"Device: {device}\n")

    # Check both checkpoint directories
    ckpt_dirs = [
        SIMCLR_CONFIG().CHECKPOINT_DIR,
        PROJECT_ROOT / "checkpoints" / "simclr_pretrain",
    ]

    for ckpt_dir in ckpt_dirs:
        if not ckpt_dir.exists():
            print(f"Skipping (not found): {ckpt_dir}")
            continue
        try:
            path, epoch = find_latest_checkpoint(ckpt_dir)
            print(f"\n{'#' * 60}")
            print(f"# Checkpoint dir: {ckpt_dir.name}  |  Latest epoch: {epoch}")
            print(f"{'#' * 60}")
            analyse(path, device)
        except FileNotFoundError as e:
            print(e)

