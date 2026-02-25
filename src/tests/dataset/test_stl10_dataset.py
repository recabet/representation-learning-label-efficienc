import unittest
from pathlib import Path

from src.data_handling.datasets import STL10Dataset
from src.configs.global_config import GLOBAL_CONFIG

from torchvision import transforms


class TestSTL10Dataset(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.raw_dir = Path(GLOBAL_CONFIG.RAW_DATA_DIR) / "stl10_binary"
        assert self.raw_dir.exists(), f"{self.raw_dir} does not exist!"

    def test_train_dataset(self):
        dataset = STL10Dataset(split="train", transform=self.transform, labeled=True)
        img, label = dataset[0]
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(img.shape[1], 96)
        self.assertEqual(img.shape[2], 96)
        self.assertIsInstance(label, int)

    def test_test_dataset(self):
        dataset = STL10Dataset(split="test", transform=self.transform, labeled=True)
        img, label = dataset[0]
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(img.shape[1], 96)
        self.assertEqual(img.shape[2], 96)
        self.assertIsInstance(label, int)

    def test_unlabeled_dataset(self):
        dataset = STL10Dataset(split="unlabeled", transform=self.transform, labeled=False)
        img = dataset[0]
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(img.shape[1], 96)
        self.assertEqual(img.shape[2], 96)



if __name__ == "__main__":
    unittest.main(exit=False)