from tqdm import tqdm
from pathlib import Path

import torch


def fit_one_epoch(model,
                  criterion,
                  optimizer,
                  data_loader,
                  device):

    model.train()
    total_loss = 0
    for x in tqdm(data_loader, desc="Training batch"):
        x1, x2 = x
        x1, x2 = x1.to(device), x2.to(device)

        h1, z1 = model(x1)
        h2, z2 = model(x2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def fit(model,
        train_loader,
        criterion,
        optimizer,
        device,
        epochs: int = 100,
        checkpoint_dir: str = "checkpoints"):

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        avg_loss = fit_one_epoch(model, criterion, optimizer, train_loader, device)
        print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"simclr_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

