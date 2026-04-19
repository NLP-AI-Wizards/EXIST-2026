import logging
import json
import os
from logging import Logger
from datetime import datetime
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary

from dataset import EXISTDataset
from datamodule import EXISTDataModule
from model import EXISTModel

logger: Logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def _build_task2_1_json_entries(ids, yes_probs, hard: bool):
    """
    Constructs the exact JSON format required by the PyEvALL submission system.
    """
    entries = []
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
                "test_case": "EXIST2025",  # DO NOT CHANGE THIS
            }
        )
    return entries


def _build_task2_2_json_entries(ids, judgemental_probs, hard: bool):
    entries = []
    for sample_id, p_judg in zip(ids, judgemental_probs):
        p_judg = float(max(0.0, min(1.0, p_judg)))
        if hard:
            value = "JUDGEMENTAL" if p_judg >= 0.5 else "DIRECT"
        else:
            value = {"JUDGEMENTAL": p_judg, "DIRECT": 1.0 - p_judg}
        entries.append({"id": str(sample_id), "value": value, "test_case": "EXIST2025"})
    return entries


def _build_task2_3_json_entries(ids, cat_probs, hard: bool):
    from dataset import TASK_2_3_CLASSES

    entries = []
    for sample_id, probs in zip(ids, cat_probs):
        if hard:
            # Multi-label hard: any class with p >= 0.5
            value = [TASK_2_3_CLASSES[i] for i, p in enumerate(probs) if p >= 0.5]
        else:
            # Multi-label soft: dictionary of all classes
            value = {TASK_2_3_CLASSES[i]: float(p) for i, p in enumerate(probs)}
        entries.append({"id": str(sample_id), "value": value, "test_case": "EXIST2025"})
    return entries


def _run_task2_1_eval(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: EXISTDataModule,
    best_ckpt: str,
    version: str,
    output_dir: str = "outputs",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, version), exist_ok=True)

    print("\n===> Running predictions on the test set via trainer.predict()...")

    # 1. Run Lightning inference.
    # This automatically loads 'best_ckpt' and iterates over 'predict_dataloader'
    predictions = trainer.predict(
        model=model,
        dataloaders=datamodule.predict_dataloader(),
        ckpt_path=best_ckpt,
    )

    # 2. Unpack the batched predictions
    all_ids = []
    all_probs_2_1 = []
    all_probs_2_2 = []
    all_probs_2_3 = []

    for batch_out in predictions:
        batch_ids = batch_out["id"]

        # Task 2.1
        probs_2_1 = torch.sigmoid(batch_out["logits_2_1"]).squeeze(-1).cpu().tolist()
        # Task 2.2
        probs_2_2 = torch.sigmoid(batch_out["logits_2_2"]).squeeze(-1).cpu().tolist()
        # Task 2.3
        probs_2_3 = torch.sigmoid(batch_out["logits_2_3"]).cpu().tolist()

        all_ids.extend(batch_ids)
        all_probs_2_1.extend(probs_2_1)
        all_probs_2_2.extend(probs_2_2)
        all_probs_2_3.extend(probs_2_3)

    # 3. Build prediction entries
    for task_name, probs_list, build_fn in [
        ("task2_1", all_probs_2_1, _build_task2_1_json_entries),
        ("task2_2", all_probs_2_2, _build_task2_2_json_entries),
        ("task2_3", all_probs_2_3, _build_task2_3_json_entries),
    ]:
        hard_entries = build_fn(all_ids, probs_list, hard=True)
        soft_entries = build_fn(all_ids, probs_list, hard=False)

        hard_path = os.path.join(output_dir, version, f"{task_name}_hard.json")
        soft_path = os.path.join(output_dir, version, f"{task_name}_soft.json")

        with open(hard_path, "w", encoding="utf-8") as f:
            json.dump(hard_entries, f, ensure_ascii=False, indent=2)

        with open(soft_path, "w", encoding="utf-8") as f:
            json.dump(soft_entries, f, ensure_ascii=False, indent=2)

        print(f"Saved {task_name} entries to: {hard_path} & {soft_path}")

    print(f"Saved HARD predictions to: {hard_path}")
    print(f"Saved SOFT predictions to: {soft_path}")


def predict(
    model_name: str,
    ckpt_path: str,
    n_samples: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    use_sensorial: Optional[bool] = None,
):
    print(f"===> Loading model from: {ckpt_path}")

    if use_sensorial is None:
        use_sensorial = model_name != "siglip"

    if model_name in {"qwen", "gemma4"}:
        use_sensorial = False

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
        n_samples=n_samples,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=False,
    )

    model = EXISTModel(model_name=model_name)

    model_dir = os.path.dirname(ckpt_path)
    version = os.path.basename(model_dir)

    _run_task2_1_eval(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        best_ckpt=ckpt_path,
        version=version + "_eval",
    )


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

    loggers: list = []
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name=version)
    loggers.append(tb_logger)

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

    if model_name in {"qwen", "gemma4"}:
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
        lr=1e-4,
        weight_decay=1e-2,
        warmup_ratio=0.3,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{version}",
        filename="{epoch:02d}",
        save_top_k=1,
        save_last=False,
        monitor="val/total_loss",
        mode="min",
    )

    model_summary = ModelSummary(max_depth=4)

    lr_monitor = LearningRateMonitor(
        logging_interval="step",
    )

    callbacks: list[Callback] = [model_checkpoint, lr_monitor, model_summary]

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

    # Generate the prediction files at the very end
    _run_task2_1_eval(
        trainer=trainer,
        model=model,
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
        required=True,
        choices=["siglip", "qwen", "gemma4", "gemini"],
        help="Model variant to train with",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to use from the dataset for quick testing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--no_sensorial",
        action="store_true",
        help="Disable physiological modality regardless of model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="EXIST2026",
        help="Weights & Biases project name for logging",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team) name for logging",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction mode instead of training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint file for prediction",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if args.predict:
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path must be provided in --predict mode")
        predict(
            model_name=args.model_name,
            ckpt_path=args.ckpt_path,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_sensorial=False if args.no_sensorial else None,
        )
    else:
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
