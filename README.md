# Representation Learning for Label Efficiency

Self-supervised pretraining with **SimCLR** on STL-10, demonstrating that
contrastive representations dramatically reduce the need for labeled data.

> **Key result:** SimCLR with only 10 % labeled data (79.3 %) beats a fully
> supervised model trained on 100 % labels (74.7 %).

---

## Project Structure

```
src/
├── analysis/               # Post-hoc analysis & visualization
│   ├── analyze_results.py  # Print experiment result tables
│   ├── check_collapse.py   # Detect representation collapse
│   ├── dataset_analysis.py # EDA on STL-10
│   ├── loss_dashboard.py   # Live Gradio loss curves
│   ├── model_analysis.py   # Architecture / param summary
│   └── visualization.py    # Confusion matrix & metric plots
├── configs/
│   ├── global_config.py    # Paths, device, seeds
│   └── simclr_config.py    # SimCLR hyper-parameters & augmentations
├── data_handling/
│   ├── datasets.py         # STL10Dataset (binary reader)
│   └── downloader.py       # Kaggle download helper
├── experiments/
│   ├── data_setup.py       # Train/test transforms, label-subset sampler
│   ├── evaluation.py       # Accuracy, F1, confusion matrix
│   ├── pretrain_simclr.py  # SimCLR pretraining entry point
│   ├── run_experiments.py  # Run all downstream experiments
│   └── set_experiment.py   # Three-way comparison orchestrator
├── losses/
│   ├── info_nce.py         # General InfoNCE loss
│   └── nt_xent.py          # NT-Xent (SimCLR contrastive loss)
├── models/
│   ├── linear_probe.py     # Single linear layer classifier
│   ├── probe_wrapper.py    # Frozen-encoder + probe wrapper
│   └── simclr.py           # SimCLR model (encoder + projection head)
├── tests/
│   └── dataset/
│       ├── test_stl10_dataset.py
│       ├── test_model_training.py
│       └── plot_images.py
├── training/
│   └── train.py            # Training loops (SimCLR + classifier)
└── utils/
    └── reproducibility.py  # Seed setting, environment logging
```

## Setup

```bash
# 1. Create environment
conda create -n label python=3.12 -y
conda activate label

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download STL-10 dataset
python -m src.data_handling.downloader
```

## Usage

### 1. Pretrain SimCLR

```bash
python -m src.experiments.pretrain_simclr
```

Trains a SimCLR encoder on the 100 K unlabeled STL-10 images.
Checkpoints are saved every 10 epochs to `checkpoints/simclr_pretrain_<model>/`.

Edit `src/configs/simclr_config.py` to change backbone, epochs, batch size,
or set `PRETRAINED = True` to initialise from ImageNet weights.

### 2. Run Downstream Experiments

```bash
python -m src.experiments.run_experiments
```

For each SimCLR checkpoint variant and label percentage (10 / 25 / 50 / 75 / 100 %),
runs a three-way comparison:

| Method | Description |
|--------|-------------|
| **Scratch** | Train ResNet from random init |
| **ImageNet** | Fine-tune ImageNet-pretrained ResNet |
| **SimCLR Probe** | Frozen SimCLR encoder + linear classifier |

Results are saved to `plots/<experiment_name>/metrics.json`.

### 3. Analyse Results

```bash
python -m src.analysis.analyze_results
```

Prints accuracy / F1 summary tables and the headline comparison.

### 4. Live Loss Dashboard (optional)

```bash
python -m src.analysis.loss_dashboard
```

Opens a Gradio dashboard at `http://localhost:7860` showing live loss
curves during pretraining.

## Results (ResNet-18, 800 epoch SimCLR)

| Labels | Scratch | ImageNet | SimCLR Probe (scratch) | SimCLR Probe (PT init) |
|-------:|--------:|---------:|-----------------------:|-----------------------:|
| 10 % | 46.0 % | 82.1 % | 79.3 % | **80.2 %** |
| 25 % | 54.8 % | 83.6 % | 81.8 % | **82.5 %** |
| 50 % | 63.7 % | 87.2 % | 83.2 % | **83.8 %** |
| 75 % | 71.4 % | 89.1 % | 84.1 % | **84.7 %** |
| 100 % | 74.7 % | 89.7 % | 84.6 % | **85.0 %** |

