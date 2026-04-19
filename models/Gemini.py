import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file

from models.model_head.mlp import ClassificationHead


def get_embeddings():
    # Load embeddings from safetensors
    train_data = load_file("data/embeddings_train.safetensors")
    test_data = load_file("data/embeddings_test.safetensors")

    # Combine embeddings
    train_ids = train_data["ids"].numpy()
    train_embeds = train_data["embeddings"]
    test_ids = test_data["ids"].numpy()
    test_embeds = test_data["embeddings"]

    all_embeddings = torch.cat([train_embeds, test_embeds], dim=0)
    all_ids = np.concatenate([train_ids, test_ids])

    id_to_embedding_idx = {int(id_val): idx for idx, id_val in enumerate(all_ids)}
    return all_embeddings, all_ids, id_to_embedding_idx


class Gemini(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        embeddings, embedding_ids, id_to_embedding_idx = get_embeddings()
        self.register_buffer("embeddings", embeddings)
        self.id_to_embedding_idx = id_to_embedding_idx

        embed_dim = self.embeddings.shape[1]

        # Using a shared projection layer for potentially better feature alignment
        self.shared_proj = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout)
        )

        self.head_2_1 = ClassificationHead(512, 1)
        self.head_2_2 = ClassificationHead(512, 1)
        self.head_2_3 = ClassificationHead(512, 5)

    def forward(self, ids: list[int]):
        """
        ids: List of sample IDs
        """
        # Map sample IDs to embedding indices
        indices = [self.id_to_embedding_idx[int(sample_id)] for sample_id in ids]
        indices = torch.tensor(indices, device=self.embeddings.device)

        # Retrieve embeddings
        features = self.embeddings[indices]

        # Shared features
        shared_features = self.shared_proj(features)

        # Output raw logits for BCEWithLogitsLoss
        logits_2_1 = self.head_2_1(shared_features)

        # Logits for Task 2.2 and 2.3
        logits_2_2 = self.head_2_2(shared_features)
        logits_2_3 = self.head_2_3(shared_features)

        # Hierarchical constraint during inference: Task 2.2 and Task 2.3
        # should technically be conditioned on Task 2.1.
        # However, for soft probabilities and training stability,
        # we return raw logits and let the loss handle conditioning
        # or apply masks where appropriate.

        return {
            "logits_2_1": logits_2_1,
            "logits_2_2": logits_2_2,
            "logits_2_3": logits_2_3,
        }
