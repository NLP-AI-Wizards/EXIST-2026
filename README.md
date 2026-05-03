# EXIST-2026: Hierarchical Sexism Identification in Memes

This repository contains the implementation for the **sEXism Identification in Social neTworks (EXIST) 2026** shared task.

---

## 🏗️ Architecture Overview

## 📂 Project Structure

- `models/Gemini.py`: Core hierarchical model implementation.
- `loss.py`: Implementation of KL Divergence and Uncertainty Weighting logic.
- `train.py`: Unified training and prediction entry point.
- `eval_results.py`: Evaluation script using `PyEvALL` to generate comparative CSV results.

---

## 🚀 Setup & Usage

### 1. Installation
Using `uv`:
```bash
uv sync
```

### 2. Training
We utilize **PyTorch Lightning** for training. The script accepts various hyperparameters for configuration and logging.
```bash
# Example training run
python train.py --model_name gemini
```

### 3. Evaluation
After training, evaluate the generated predictions:
```bash
python eval_results.py
```

---

## 📜 Citation
If you find this implementation useful, please consider citing our work:
```bibtex
```

## 📄 License
MIT License
