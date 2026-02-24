from src.data_handling.datasets import STL10Dataset

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_images():
    """Plot 3 images from each split"""
    transform = transforms.Compose([])  # no tensor conversion for plotting
    splits = ["train", "test", "unlabeled"]

    for split in splits:
        dataset = STL10Dataset(split=split, transform=transform, labeled=(split != "unlabeled"))
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        fig.suptitle(f"{split} samples")
        for i in range(3):
            if split == "unlabeled":
                img = dataset[i]
            else:
                img, label = dataset[i]
            axes[i].imshow(np.array(img))
            axes[i].axis("off")
        plt.show()


if __name__ == "__main__":
    plot_sample_images()