# AI Wizards at EXIST 2026

Public release of the AI Wizards code for the EXIST 2026 shared task on hierarchical sexism identification in memes.

The final system uses fixed Gemini Embedding 2 meme representations and lightweight task heads to predict the three EXIST levels jointly with soft-label supervision and hierarchical decoding.

## Repository Layout

- `train.py`: training entry point that builds the split, trains the model, and exports prediction JSON files.
- `predict.py`: checkpoint-only inference entry point for official-format submissions.
- `model.py`: Lightning wrapper with training, validation, prediction, optimizer, and metrics.
- `models/Gemini.py`: fixed-embedding Gemini model and optional fusion modules.
- `models/SigLIP.py`: frozen SigLIP baseline.
- `dataset.py`: EXIST JSON loader and soft-target construction.
- `datamodule.py`: Lightning datamodule, split logic, and batch collation.
- `loss.py`: custom loss functions and uncertainty weighting.
- `eval_results.py`: local evaluation helper for generated JSON files.
- `exist2026_aiwizards/`: official-format submission files.

## Data

The code expects the official EXIST data under `data/`:

```text
data/
  EXIST 2026 Memes Dataset/
    training/EXIST2026_training.json
    training/memes/
    test/EXIST2026_test_clean.json
    test/memes/
```

The Gemini embedding `.safetensors` files are **not** included in this repository. They will be provided on request by opening an issue.

Optional demographic embeddings, if used for ablations, follow the same convention:

```text
data/embeddings_train_demographics.safetensors
data/embeddings_test_demographics.safetensors
```

## Installation

```bash
uv sync
```

The project targets Python 3.12.

## Training

```bash
uv run python train.py \
  --model_name gemini \
  --version final \
  --epochs 50 \
  --soft_gating
```

Useful flags:

```bash
--use_demographics   # load demographic-augmented cached embeddings
--use_sensorial      # enable physiological feature loading and fusion
--use_subject_ids    # condition physiological features on subject IDs
--model_name siglip  # train the frozen SigLIP baseline instead of Gemini embeddings
--wandb              # enable Weights & Biases logging
```

Training writes checkpoints to `checkpoints/<model>_<version>/`, TensorBoard logs to `tb_logs/`, and prediction JSON files to `outputs/<model>_<version>/`.

## Prediction

```bash
uv run python predict.py \
  --ckpt_path checkpoints/gemini_final/epoch=XX.ckpt \
  --version gemini_final \
  --output_dir exist2026_aiwizards
```

This produces hard and soft JSON files for Task 2.1, Task 2.2, and Task 2.3.

## Evaluation

```bash
uv run python eval_results.py \
  --predictions_files outputs/gemini_final/task2_1_soft_aiwizards_1.json \
  --task 2_1 \
  --mode soft
```

If `--gold_path` is omitted, the script uses the local gold files under `data/evaluation/golds/`.

## Citation

If you use this repository, please cite:

```bibtex
@misc{fasulo2026aiwizardsexist2026,
      title={AI Wizards at EXIST 2026: Hierarchical Soft-Label Learning for Multimodal Sexism Identification in Memes},
      author={Matteo Fasulo and Antonio Gravina and Luca Tedeschini and Luca Babboni},
      year={2026},
      eprint={2607.04410},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.04410},
}
```

## License

See [LICENSE](LICENSE). The repository is released under CC BY 4.0.
