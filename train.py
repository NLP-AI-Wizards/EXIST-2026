import logging
import json
import os
from logging import Logger
from datetime import datetime

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

        entries.append({
            "id": str(sample_id),
            "value": value,
            "test_case": "EXIST2025",  # DO NOT CHANGE THIS
        })
    return entries


def _build_task2_2_json_entries(ids, p_2_1, judgemental_probs, hard: bool):
    entries = []
    for sample_id, p_yes, p_judg in zip(ids, p_2_1, judgemental_probs):
        p_yes = float(max(0.0, min(1.0, p_yes)))
        p_judg = float(max(0.0, min(1.0, p_judg)))

        if hard:
            if p_yes < 0.5:
                value = "NO"
            else:
                value = "JUDGEMENTAL" if p_judg >= 0.5 else "DIRECT"
        else:
            # Task 2.2 soft labels: DIRECT, JUDGEMENTAL, NO
            # TODO: (Check if correct) Redistribute probabilities so they sum to 1 and reflect hierarchy
            if p_yes >= 0.5:
                value = {
                    "JUDGEMENTAL": p_judg,
                    "DIRECT": 1.0 - p_judg,
                    "NO": 0.0,
                }
            else:
                value = {
                    "JUDGEMENTAL": 0.0,
                    "DIRECT": 0.0,
                    "NO": 1.0,
                }
        entries.append({"id": str(sample_id), "value": value, "test_case": "EXIST2025"})
    return entries


def _build_task2_3_json_entries(ids, p_2_1, cat_probs, hard: bool):
    from dataset import TASK_2_3_CLASSES

    entries = []
    for sample_id, p_yes, probs in zip(ids, p_2_1, cat_probs):
        p_yes = float(max(0.0, min(1.0, p_yes)))
        if hard:
            if p_yes < 0.5:
                value = ["NO"]
            else:
                # Multi-label hard: any class with p >= 0.5
                value = [TASK_2_3_CLASSES[i] for i, p in enumerate(probs) if p >= 0.5]
                if not value:  # Fallback if p_yes >= 0.5 but no class >= 0.5
                    # Maybe pick the highest one or just NO? Usually if p_yes >= 0.5 there should be one.
                    # But for consistency with golden files, let's keep it as is or add NO if empty.
                    value = ["NO"]
        else:
            # Multi-label soft: dictionary of all classes + NO
            if p_yes >= 0.5:
                sum_p = sum(probs)
                if sum_p > 0:
                    normalized = [float(p) / sum_p for p in probs]
                else:
                    normalized = [1.0 / len(probs)] * len(probs)

                value = {TASK_2_3_CLASSES[i]: p for i, p in enumerate(normalized)}
                value["NO"] = 0.0
            else:
                value = {TASK_2_3_CLASSES[i]: 0.0 for i in range(len(probs))}
                value["NO"] = 1.0

        entries.append({"id": str(sample_id), "value": value, "test_case": "EXIST2025"})
    return entries


def _run_eval(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: EXISTDataModule,
    best_ckpt: str,
    version: str,
    output_dir: str = "outputs",
):
    if args.paper_run:
        output_dir = output_dir + "_paper"

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

    # 3. Zip all data together and sort by ID (numeric sort for string IDs)
    # We use key=lambda x: int(x[0]) because sample IDs are strings of integers
    combined = list(zip(all_ids, all_probs_2_1, all_probs_2_2, all_probs_2_3))
    combined.sort(key=lambda x: int(x[0]))

    # Unpack sorted data
    sorted_ids, sorted_probs_2_1, sorted_probs_2_2, sorted_probs_2_3 = zip(*combined)

    # 4. Build prediction entries
    for task_name, probs_list, build_fn in [
        (
            "task2_1",
            sorted_probs_2_1,
            lambda ids, probs, hard: _build_task2_1_json_entries(ids, probs, hard),
        ),
        (
            "task2_2",
            sorted_probs_2_2,
            lambda ids, probs, hard: _build_task2_2_json_entries(
                ids, sorted_probs_2_1, probs, hard
            ),
        ),
        (
            "task2_3",
            sorted_probs_2_3,
            lambda ids, probs, hard: _build_task2_3_json_entries(
                ids, sorted_probs_2_1, probs, hard
            ),
        ),
    ]:
        hard_entries = build_fn(sorted_ids, probs_list, hard=True)
        soft_entries = build_fn(sorted_ids, probs_list, hard=False)

        hard_path = os.path.join(output_dir, version, f"{task_name}_hard.json")
        soft_path = os.path.join(output_dir, version, f"{task_name}_soft.json")

        with open(hard_path, "w", encoding="utf-8") as f:
            json.dump(hard_entries, f, ensure_ascii=False, indent=2)

        with open(soft_path, "w", encoding="utf-8") as f:
            json.dump(soft_entries, f, ensure_ascii=False, indent=2)

        print(f"Saved {task_name} entries to: {hard_path} & {soft_path}")

    print(f"Saved HARD predictions to: {hard_path}")
    print(f"Saved SOFT predictions to: {soft_path}")


def train(seed):
    if args.paper_run:
        version = f"{args.model_name}_{args.version}_seed_{seed}"
    else:
        version = f"{args.model_name}_{args.version}"


    loggers: list = []
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name=version)
    loggers.append(tb_logger)

    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=version,
            save_dir="wandb",
            entity=args.wandb_entity,
        )
        loggers.append(wandb_logger)

    # DataLoader
    print("===> Loading datasets")
    datamodule = EXISTDataModule(
        train_dataset=EXISTDataset(
            json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
            img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
            use_sensorial=args.use_sensorial,
        ),
        test_dataset=EXISTDataset(
            json_path=r"data/EXIST 2026 Memes Dataset/test/EXIST2026_test_clean.json",
            img_dir=r"data/EXIST 2026 Memes Dataset/test/memes",
            use_sensorial=args.use_sensorial,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Pytorch Lightning module
    print("===> Start building model")
    model = EXISTModel(
        model_name=args.model_name,
        n_blocks=args.n_blocks,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        soft_gating=args.soft_gating
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

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val/total_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    callbacks: list[Callback] = [
        model_checkpoint,
        lr_monitor,
        model_summary,
        early_stopping,
    ]

    # Trainer
    print("===> Instantiate trainer")
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
    )

    print("===> Start training")
    trainer.fit(model, datamodule)

    best_ckpt = model_checkpoint.best_model_path

    print(f"Best checkpoint path: {best_ckpt}")

    # Generate the prediction files at the very end
    _run_eval(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        best_ckpt=best_ckpt,
        version=version,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EXIST2026 Model")

    # MODEL
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["gemini"],
        help="Model variant to train with",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=datetime.now().strftime("%d_%m_%H-%M"),
        help="Version name for logging and checkpointing (default: timestamp)",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="Number of expansion blocks in Gemini",
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=2,
        help="Expansion factor for the Gemini blocks",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for the Gemini blocks",
    )

    parser.add_argument(
        "--soft_gating",
        action="store_true",
        help="Add soft gating to model architecture",
    )


    # OPTIMIZATION
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.3,
        help="Warmup ratio for learning rate scheduling",
    )

    # TRAINING
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--use_sensorial",
        action="store_true",
        help="Enable physiological modality",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--paper_run",
        action="store_true",
        help="Train the model over five different seeds, then averages the results",
    )

    # LOGGING
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
    args = parser.parse_args()

    seeds = [42,67,89,110,1337]
    if args.paper_run:
        for seed in seeds:
            pl.seed_everything(seed)
            train(seed)
    
    else:
        # Set random seed for reproducibility
        pl.seed_everything(args.seed)

        # Start training
        train(0)
