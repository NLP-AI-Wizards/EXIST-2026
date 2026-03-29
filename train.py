import logging
from logging import Logger
from datetime import datetime

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset import EXISTDataset
from datamodule import EXISTDataModule
from model import EXISTModel

logger: Logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

def train(
    model_name: str = "discern_tiny",
    epochs: int = 10,
    n_samples: int = None,
    batch_size: int = 16,
    num_workers: int = 4,
    wandb_project: str = "EXIST2026",
    use_wandb: bool = False,
):
    date_format = "%d_%m_%H-%M"

    version = f"{model_name}_{datetime.now().strftime(date_format)}"

    # TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="EXIST2026",
        version=version,
    )
    loggers = [tb_logger]

    if use_wandb:
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=version,
            log_model="all",
            save_dir="wandb",
        )
        loggers.append(wandb_logger)

    # DataLoader
    print("===> Loading datasets")
    datamodule = EXISTDataModule(
        dataset=EXISTDataset(
            json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
            img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
            use_sensorial=True,
            use_annotator_metadata=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            n_samples=n_samples,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Pytorch Lightning module
    print("===> Start building model")
    model = EXISTModel(model_name=model_name)
    print(model)

    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{version}",
        filename="{epoch:02d}",
        save_top_k=1,
        save_last=True,
        monitor="val/total_loss",
        mode="min",
    )

    model_summary = pl.callbacks.ModelSummary(max_depth=4)
    callbacks = [model_checkpoint, model_summary]

    # Trainer
    print("===> Instantiate trainer")
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16" if torch.cuda.is_available() else "32",
        gradient_clip_val=1.0,
    )

    print("===> Start training")
    trainer.fit(model, datamodule)

    best_ckpt = model_checkpoint.best_model_path

    print(f"Best checkpoint path: {best_ckpt}")
    print(f"Best model score: {model_checkpoint.best_model_score}")

    trainer.validate(model, datamodule, ckpt_path=best_ckpt)

    trainer.test(model, datamodule, ckpt_path=best_ckpt)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EXIST2026 Model")
    parser.add_argument("--model_name", type=str, default="discern_tiny", help="Model variant to train (discern_tiny, discern_base, discern_large)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use from the dataset for training (for quick testing)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--wandb_project", type=str, default="EXIST2026", help="Weights & Biases project name for logging")
    parser.add_argument("--wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        use_wandb=args.wandb,
    )