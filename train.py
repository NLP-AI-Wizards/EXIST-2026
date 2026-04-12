import logging
import json
import os
from logging import Logger
from datetime import datetime
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset import EXISTDataset
from datamodule import EXISTDataModule
from model import EXISTModel

logger: Logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def _build_task2_1_json_entries(ids, yes_probs, hard: bool):
    """
    Constructs the exact JSON format required by the PyEvALL submission system.
    """
    entries =[]
    for sample_id, p_yes in zip(ids, yes_probs):
        # Clip probabilities for safety
        p_yes = float(max(0.0, min(1.0, p_yes)))

        if hard:
            # Majority vote
            value = "YES" if p_yes >= 0.5 else "NO"
        else:
            # Probabilistic dict
            value = {"YES": p_yes, "NO": 1.0 - p_yes}

        entries.append(
            {
                "id": str(sample_id),
                "value": value,
                "test_case": "EXIST2025", # DO NOT CHANGE THIS
            }
        )
    return entries


def _run_task2_1_eval(
    trainer: pl.Trainer,
    datamodule: EXISTDataModule,
    best_ckpt: str,
    version: str,
    output_dir: str = "outputs",
):
    if not best_ckpt:
        print("No best checkpoint available. Skipping PyEvALL evaluation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\n===> Running predictions on the test set via trainer.predict()...")

    # 1. Run Lightning inference.
    # This automatically loads 'best_ckpt' and iterates over 'predict_dataloader'
    predictions = trainer.predict(
        model=None, # None forces it to load from ckpt_path
        dataloaders=datamodule.predict_dataloader(), # TODO: set to be training loader to assess 2025 perf, needs to be changed to test loader for 2026 eval
        ckpt_path=best_ckpt
    )

    # 2. Unpack the batched predictions
    all_ids = []
    all_probs_yes =[]

    for batch_out in predictions:
        batch_ids = batch_out["id"]

        # Extract logits and apply sigmoid to get [0, 1] marginal probabilities
        logits_2_1 = batch_out["logits_2_1"]
        probs = torch.sigmoid(logits_2_1).squeeze(-1).cpu().tolist()

        all_ids.extend(batch_ids)
        all_probs_yes.extend(probs)

    # 3. Build Hard and Soft JSON configurations
    hard_entries = _build_task2_1_json_entries(all_ids, all_probs_yes, hard=True)
    soft_entries = _build_task2_1_json_entries(all_ids, all_probs_yes, hard=False)

    hard_path = os.path.join(output_dir, f"{version}_task2_1_hard.json")
    soft_path = os.path.join(output_dir, f"{version}_task2_1_soft.json")

    # 4. Save to disk
    with open(hard_path, "w", encoding="utf-8") as f:
        json.dump(hard_entries, f, ensure_ascii=False, indent=2)

    with open(soft_path, "w", encoding="utf-8") as f:
        json.dump(soft_entries, f, ensure_ascii=False, indent=2)

    print(f"Saved HARD predictions to: {hard_path}")
    print(f"Saved SOFT predictions to: {soft_path}")


def train(
    model_name: str,
    epochs: int = 5,
    n_samples: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    use_sensorial: Optional[bool] = None,
    wandb_project: str = "EXIST2026",
    use_wandb: bool = False,
    wandb_entity: Optional[str] = None,
):
    date_format = "%d_%m_%H-%M"
    version = f"{model_name}_{datetime.now().strftime(date_format)}"

    loggers: list =[]

    if use_wandb:
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=version,
            save_dir="wandb",
            entity=wandb_entity,
        )
        loggers.append(wandb_logger)

    if use_sensorial is None:
        use_sensorial = model_name != "siglip"

    if model_name == "qwen":
        use_sensorial = False

    # DataLoader
    print("===> Loading datasets")
    datamodule = EXISTDataModule(
        train_dataset=EXISTDataset(
            json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
            img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
            use_sensorial=use_sensorial,
        ),
        test_dataset=EXISTDataset(
            json_path=r"data/EXIST 2026 Memes Dataset/test/EXIST2026_test_clean.json",
            img_dir=r"data/EXIST 2026 Memes Dataset/test/memes",
            use_sensorial=use_sensorial,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        n_samples=n_samples,
    )

    # Pytorch Lightning module
    print("===> Start building model")
    model = EXISTModel(
        model_name=model_name,
        lr=5e-4,
        weight_decay=1e-2,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{version}",
        filename="{epoch:02d}",
        save_top_k=1,
        save_last=True,
        monitor="val/total_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
        log_weight_decay=True,
    )

    callbacks: list[Callback] = [model_checkpoint, lr_monitor]

    # Trainer
    print("===> Instantiate trainer")
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
    )

    print("===> Start training")
    trainer.fit(model, datamodule)

    best_ckpt = model_checkpoint.best_model_path

    print(f"Best checkpoint path: {best_ckpt}")
    print(f"Best model score: {model_checkpoint.best_model_score}")

    # Generate the prediction files at the very end
    _run_task2_1_eval(
        trainer=trainer,
        datamodule=datamodule,
        best_ckpt=best_ckpt,
        version=version,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EXIST2026 Model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="siglip",
        choices=["siglip", "qwen"],
        help="Model variant to train with",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use from the dataset for quick testing")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--no_sensorial", action="store_true", help="Disable physiological modality regardless of model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="EXIST2026", help="Weights & Biases project name for logging")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team) name for logging")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_sensorial=False if args.no_sensorial else None,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )