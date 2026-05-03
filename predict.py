import argparse
import os

import pytorch_lightning as pl

from datamodule import EXISTDataModule
from dataset import EXISTDataset
from model import EXISTModel
from train import _get_model_input_flags, _run_eval


def generate_predictions(args):
    print(f"===> Loading model from checkpoint: {args.ckpt_path}")

    # Load the model directly from the checkpoint (with hyperparameters)
    model = EXISTModel.load_from_checkpoint(args.ckpt_path)  # use_subject_ids=True
    print(model)
    model.eval()

    input_flags = _get_model_input_flags(model.hparams.model_name)

    print(f"===> Loading evaluation dataset from: {args.eval_json}")
    datamodule = EXISTDataModule(
        test_dataset=EXISTDataset(
            json_path=args.eval_json,
            img_dir=args.eval_img_dir,
            include_image=input_flags["include_image"],
            include_text=input_flags["include_text"],
            include_id=input_flags["include_id"],
            use_sensorial=model.hparams.use_sensorial,
        ),
        batch_size=8,
        num_workers=4,
        include_image=input_flags["include_image"],
        include_text=input_flags["include_text"],
        include_id=input_flags["include_id"],
    )

    datamodule.setup(stage="predict")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=False,
    )

    _run_eval(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        best_ckpt=args.ckpt_path,
        version="",
        output_dir=os.path.join(args.output_dir, args.version),
    )

    print("===> Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate EXIST Predictions from Checkpoint"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the trained PyTorch Lightning .ckpt file",
    )

    parser.add_argument(
        "--eval_json",
        type=str,
        default=r"data/EXIST 2026 Memes Dataset/test/EXIST2026_test_clean.json",
        help="Path to the JSON file you want to evaluate on",
    )
    parser.add_argument(
        "--eval_img_dir",
        type=str,
        default=r"data/EXIST 2026 Memes Dataset/test/memes",
        help="Path to the images directory you want to evaluate on",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="exist2026_aiwizards",
        help="Directory to save the generated JSON files",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Subfolder/prefix name for the output files",
    )
    args = parser.parse_args()

    generate_predictions(args)
