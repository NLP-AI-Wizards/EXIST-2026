import argparse
import os

import pytorch_lightning as pl

from dataset import EXISTDataset
from datamodule import EXISTDataModule
from model import EXISTModel
from train import _run_eval, _get_model_input_flags


def generate_predictions(args):
    print(f"===> Loading model from checkpoint: {args.ckpt_path}")

    # 1. Load the model directly from the checkpoint
    model = EXISTModel.load_from_checkpoint(args.ckpt_path)
    model.eval()

    model_name = model.hparams.model_name
    input_flags = _get_model_input_flags(model_name)

    # 2. Setup the DataModule
    # TRICK: We pass the TARGET evaluation dataset (train/val/test) into the `test_dataset` slot
    # so that `trainer.predict()` runs over it smoothly.
    print(f"===> Loading evaluation dataset from: {args.eval_json}")
    datamodule = EXISTDataModule(
        train_dataset=EXISTDataset(
            json_path=args.eval_json,
            img_dir=args.eval_img_dir,
            include_image=input_flags["include_image"],
            include_text=input_flags["include_text"],
            include_id=input_flags["include_id"],
            use_sensorial=args.use_sensorial,
            # If you want to test on a small subset, pass n_samples=args.n_samples here
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_image=input_flags["include_image"],
        include_text=input_flags["include_text"],
        include_id=input_flags["include_id"],
    )

    datamodule.setup(stage="predict")

    # 3. Instantiate a barebones Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=False, # Disable wandb/tensorboard for clean inference
    )

    # 4. Generate the PyEvALL JSON Entries
    version_name = args.version
    if not version_name:
        # Match training run naming by defaulting to the checkpoint parent directory.
        version_name = os.path.basename(os.path.dirname(os.path.normpath(args.ckpt_path)))
        if not version_name:
            version_name = f"{model_name}_train_eval"

    _run_eval(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        best_ckpt=args.ckpt_path,
        version=version_name,
        output_dir=args.output_dir
    )

    print("===> Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EXIST Predictions from Checkpoint")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the trained PyTorch Lightning .ckpt file"
    )

    parser.add_argument(
        "--eval_json",
        type=str,
        default=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
        help="Path to the JSON file you want to evaluate on"
    )
    parser.add_argument(
        "--eval_img_dir",
        type=str,
        default=r"data/EXIST 2026 Memes Dataset/training/memes",
        help="Path to the images directory you want to evaluate on"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the generated JSON files"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Subfolder/prefix name for the output files"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--use_sensorial", action="store_true", help="Enable physiological modality")

    args = parser.parse_args()

    generate_predictions(args)