import os
import zipfile
from pathlib import Path
from src.configs.global_config import GLOBAL_CONFIG


def download_stl10_from_kaggle():
    """
    Downloads the binary STL-10 dataset from Kaggle (pratt3000/stl10-binary-files),
    extracts it into data/raw/, and deletes the zip file.
    """
    raw_dir = Path(GLOBAL_CONFIG.RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = "pratt3000/stl10-binary-files"

    print("Downloading STL-10 binary files from Kaggle...")
    os.system(f"kaggle datasets download -d {dataset} -p {raw_dir} --unzip")

    zip_path = raw_dir / "stl10-binary-files.zip"
    if zip_path.exists():
        print("Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()

    print("STL-10 downloaded and extracted to:", raw_dir)


if __name__ == "__main__":
    download_stl10_from_kaggle()