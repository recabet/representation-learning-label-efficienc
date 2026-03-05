from src.experiments.set_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(experiment_name="10",
                   percent_labels=10,
                   simclr_path="/home/recabet/representation-learning-label-efficiency/checkpoints/simclr_pretrain/simclr_epoch_500.pth",
                   batch_size=64,
                   epochs=100,
                   device="cuda")

    run_experiment(experiment_name="25",
                   percent_labels=25,
                   simclr_path="/home/recabet/representation-learning-label-efficiency/checkpoints/simclr_pretrain/simclr_epoch_500.pth",
                   batch_size=64,
                   epochs=100,
                   device="cuda")
    run_experiment(experiment_name="50",
                   percent_labels=50,
                   simclr_path="/home/recabet/representation-learning-label-efficiency/checkpoints/simclr_pretrain/simclr_epoch_500.pth",
                   batch_size=64,
                   epochs=100,
                   device="cuda")
    run_experiment(experiment_name="75",
                   percent_labels=75,
                   simclr_path="/home/recabet/representation-learning-label-efficiency/checkpoints/simclr_pretrain/simclr_epoch_500.pth",
                   batch_size=64,
                   epochs=100,
                   device="cuda")
    run_experiment(experiment_name="100",
                   percent_labels=100,
                   simclr_path="/home/recabet/representation-learning-label-efficiency/checkpoints/simclr_pretrain/simclr_epoch_500.pth",
                   batch_size=64,
                   epochs=100,
                   device="cuda")
