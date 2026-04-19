# EXIST-2026: Hierarchical Sexism Identification in Memes

This repository contains the implementation for the **sEXism Identification in Social neTworks (EXIST) 2026** shared task. Our approach as the **AI Wizards** team focuses on leveraging high-dimensional vision-language embeddings with a hierarchical, multi-task architecture to jointly solve the three sub-tasks of the competition.

---

## 🏗️ Architecture Overview

The proposed architecture handles the complexity of jointly learning three related tasks (sexist classification, intention, and category) while maintaining a shared representation space.

### 1. Multi-Task Learning Architecture
The proposed architecture handles the complexity of jointly learning three related tasks (sexist classification, intention, and category) while maintaining a shared representation space.
- **Shared Projection**: A unified backbone processes the input embeddings into a shared feature space for all sub-tasks.
- **Independent Task Heads**: Dedicated classification heads emerge from the shared representation to handle the specific requirements of Task 2.1, 2.2, and 2.3.

### 2. Multi-Task Learning with Homoscedastic Uncertainty
Balancing losses across tasks of varying difficulty (binary vs. multi-label) is notoriously challenging. We implement the multi-task loss weighting proposed by **Kendall et al. (CVPR 2018)**, which derives the objective by maximizing the Gaussian likelihood of the multi-task output with homoscedastic task uncertainty.

For a set of $T$ tasks with individual losses $\mathcal{L}_i$, the objective minimizes:

$$
\mathcal{L}_{total} \approx \sum_{i=1}^{T} \left( \frac{1}{\sigma_i^2} \mathcal{L}_i + \log \sigma_i \right)
$$

To ensure numerical stability and avoid division by zero during optimization, we parameterize the variance as $s_i = \log(\sigma_i^2)$. Substituting this yields our implemented loss function:

$$
\mathcal{L}_{total} = \sum_{i=1}^{T} \left( e^{-s_i} \mathcal{L}_i + \frac{1}{2} s_i \right)
$$

By jointly learning the network parameters and $s_i$, the model balances the tasks dynamically according to their learned relative noise levels.

### 3. KL-Divergence vs BCE in Soft Target Scenarios
Under soft-target scenarios (where target $p \in (0,1)$), using Binary Cross Entropy (BCE) is **not equivalent** to KL Divergence when combined with uncertainty weighting.

The mathematical relationship between the two is:

$$
D_{KL}(p \parallel q) = p \log \frac{p}{q} + (1-p)\log\frac{1-p}{1-q} = \text{BCE}(p, q) - \mathcal{H}(p)
$$

where $\mathcal{H}(p) = - \left[ p \log p + (1-p) \log (1-p) \right]$ is the entropy of the annotators' target distribution.

While $\nabla_{\theta}\text{BCE} = \nabla_{\theta}D_{KL}$ for the shared neural network weights $\theta$, the gradients for the uncertainty parameters $s_i$ are fundamentally distorted by BCE. With BCE, the minimum attainable loss is lower-bounded by the dataset's entropy ($\mathcal{L}_{min} = \mathcal{H}(p) > 0$). When computing the derivative with respect to the uncertainty scale:

$$
\frac{\partial \mathcal{L}_{total}}{\partial s_i} = - e^{-s_i} \mathcal{L}_i + \frac{1}{2}
$$

an irreducible positive $\mathcal{L}_i$ artificially inflates the learned variance $s_i$. This causes the model to erroneously down-weight tasks with high label entropy, treating the annotators' disagreement as predictive failure. By directly optimizing $D_{KL}$, we ensure the theoretical global minimum of the loss is exactly $0$. This allows the homoscedastic scale $e^{-s_i}$ to correctly track the model's true predictive divergence rather than the intrinsic noise of soft annotations.

### 4. SwiGLU Shared Backbone
The shared projection layer utilizes **Expansion Blocks** featuring:
- **SwiGLU Activation**: High-performance activation used in modern LLMs (like Llama) for superior non-linear feature transformation.
- **LayerNorm & Residuals**: Ensures stable gradient flow through the shared embedding space.

---

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
uv venv .venv
source .venv/bin/activate
uv sync
```

### 2. Training
We utilize **PyTorch Lightning** for training. The script accepts various hyperparameters for configuration and logging.
```bash
# Example training run
python train.py --model_name gemini --epochs 15 --batch_size 16
```

### 3. Evaluation
After training, evaluate the generated predictions across all tasks and modes:
```bash
python eval_results.py --all
```
This generates `evaluation_results_hard.csv` and `evaluation_results_soft.csv` for detailed side-by-side comparison.

---

## 📊 Evaluation Metrics
*TBD - Detailed results analysis forthcoming.*

## 📜 Citation
If you find this implementation useful, please consider citing our work:
```bibtex
```

## 📄 License
MIT License
