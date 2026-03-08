from src.experiments.set_experiment import run_experiment
from src.configs.global_config import GLOBAL_CONFIG

if __name__ == "__main__":
    # ─────────────────────────────────────────────
    # SimCLR checkpoint variants: (name, architecture, checkpoint_path, out_dim)
    # ─────────────────────────────────────────────
    variants = [
        ("resnet18",
         "resnet18",
         f"{GLOBAL_CONFIG.SAVE_DIR}/simclr_pretrain_resnet18/simclr_epoch_800.pth",
         128),
        ("resnet18_pt",
         "resnet18",
         f"{GLOBAL_CONFIG.SAVE_DIR}/simclr_pretrain_resnet18_pt/simclr_epoch_800.pth",
         128),
        # ("resnet50",
        #  "resnet50",
        #  f"{GLOBAL_CONFIG.SAVE_DIR}/simclr_pretrain_resnet50/simclr_epoch_800.pth",
        #  256),
    ]

    label_percents = [10, 25, 50, 75, 100]

    for variant_name, arch, simclr_path, out_dim in variants:
        for pct in label_percents:
            run_experiment(
                experiment_name=f"{variant_name}_{pct}",
                percent_labels=pct,
                simclr_path=simclr_path,
                batch_size=64,
                epochs=100,
                device="cuda",
                base_model=arch,
                out_dim=out_dim,
            )
