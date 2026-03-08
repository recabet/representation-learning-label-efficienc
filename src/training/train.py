from tqdm import tqdm
from pathlib import Path
import csv
import time
import os

import torch
from src.configs.global_config import GLOBAL_CONFIG

# Cross-platform file locking
if os.name == "nt":
    import msvcrt
    def _lock(f):   msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def _unlock(f): msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl
    def _lock(f):   fcntl.flock(f, fcntl.LOCK_EX)
    def _unlock(f): fcntl.flock(f, fcntl.LOCK_UN)


def _append_loss_csv(csv_path: Path, epoch: int, loss: float, lr: float):
    """Append one row to the loss log CSV (creates header if file is new).
    Uses a file lock so parallel processes writing to the same CSV don't corrupt it."""
    lock_path = csv_path.with_suffix(".csv.lock")
    write_header = not csv_path.exists()

    with open(lock_path, "w") as lock_f:
        _lock(lock_f)
        try:
            # Re-check after acquiring lock (another process may have created it)
            if write_header and csv_path.exists():
                write_header = False
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "loss", "lr", "timestamp"])
                writer.writerow([epoch, f"{loss:.6f}", f"{lr:.8f}", time.time()])
        finally:
            _unlock(lock_f)


def fit_one_epoch_simclr(model,
                         criterion,
                         optimizer,
                         data_loader,
                         device,
                         scaler=None):

    model.train()
    total_loss = 0
    use_amp = scaler is not None

    for x in tqdm(data_loader, desc="Training batch"):
        x1, x2 = x
        x1, x2 = x1.to(device), x2.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            h1, z1 = model(x1)
            h2, z2 = model(x2)
            loss = criterion(z1, z2)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss



def fit_one_epoch_cls(model,
                      data_loader,
                      criterion,
                      optimizer,
                      device):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(data_loader, desc="Training batch"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy

def fit_simclr(model,
               train_loader,
               criterion,
               optimizer,
               device,
               epochs: int = 100,
               checkpoint_dir: str = "checkpoints",
               scheduler=None):

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_csv = checkpoint_dir / "loss_log.csv"

    # AMP scaler for mixed-precision training
    use_amp = GLOBAL_CONFIG.USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for epoch in range(1, epochs + 1):
        avg_loss = fit_one_epoch_simclr(model, criterion, optimizer,
                                        train_loader, device, scaler=scaler)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f}  LR: {current_lr:.6f}")

        _append_loss_csv(loss_csv, epoch, avg_loss, current_lr)

        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"simclr_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


def fit_cls(model,
            train_loader,
            device,
            epochs: int = 100,
            lr: float = 1e-3,
            checkpoint_dir: str = "checkpoints_cls",
            scheduler=None,
            use_sgd: bool = False):

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_csv = checkpoint_dir / "loss_log.csv"

    for epoch in range(1, epochs + 1):

        avg_loss, acc = fit_one_epoch_cls(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch}/{epochs}] "
              f"Loss: {avg_loss:.4f}  "
              f"Acc: {acc:.4f}  "
              f"LR: {current_lr:.6f}")

        _append_loss_csv(loss_csv, epoch, avg_loss, current_lr)

        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"classifier_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    return model
