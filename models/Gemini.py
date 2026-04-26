import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file

from models.model_head.mlp import ClassificationHead, SwiGLU


def get_embeddings(
    use_demographics: bool = False,
):
    if use_demographics:
        # Load embeddings from safetensors
        train_data = load_file("data/embeddings_train_demographics.safetensors")
        test_data = load_file("data/embeddings_test_demographics.safetensors")
    else:
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


class ExpansionBlock(nn.Module):
    def __init__(self, dim_in, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in, eps=1e-6)
        hidden_dim = dim_in * expansion_factor
        self.swiglu = SwiGLU(dim_in, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim_in, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x + residual


class BiosignalCrossAttention(nn.Module):
    def __init__(self, semantic_dim=768, physio_dim=108):
        super().__init__()
        self.physio_proj = nn.Linear(physio_dim, semantic_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=semantic_dim // 64,  # QKV dim: 64
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(semantic_dim)

    def forward(self, V_semantic, physio_features, physio_mask):
        """
        V_semantic: (Batch, Semantic_Dim)
        physio_features: (Batch, Num_Subjects, 108)
        physio_mask: (Batch, Num_Subjects)
        """
        # Sanitize JSON Nulls (prevents NaN propagation)
        physio_features = torch.nan_to_num(physio_features, nan=0.0)

        V_physio = self.physio_proj(physio_features)  # (B, S, Dim)

        # Semantic vector acts as a single Query token -> (B, 1, Dim)
        query = V_semantic.unsqueeze(1)

        attn_padding_mask = ~physio_mask.bool()

        # Prevent NaN crash for memes with no physio data ---
        all_masked = attn_padding_mask.all(dim=-1)
        safe_mask = attn_padding_mask.clone()
        safe_mask[all_masked, 0] = False

        attended_physio, _ = self.cross_attn(
            query=query,
            key=V_physio,
            value=V_physio,
            key_padding_mask=safe_mask,
        )

        attended_physio = attended_physio.squeeze(1)

        # Force the attended output to be exactly 0.0 for memes without physio data
        attended_physio[all_masked] = 0.0

        # Residual fusion (Pre-Norm)
        return V_semantic + self.norm(attended_physio)


class Gemini(nn.Module):
    def __init__(
        self,
        n_blocks: int = 2,
        expansion_factor: int = 4,
        dropout: float = 0.2,
        soft_gating: bool = False,
        use_demographics: bool = False,
        use_sensorial: bool = False,
    ):
        super().__init__()
        embeddings, embedding_ids, id_to_embedding_idx = get_embeddings(
            use_demographics=use_demographics
        )
        self.register_buffer("embeddings", embeddings)
        self.id_to_embedding_idx = id_to_embedding_idx
        self.soft_gating = soft_gating
        self.use_sensorial = use_sensorial

        embed_dim = self.embeddings.shape[1]

        if self.use_sensorial:
            self.physio_fusion = BiosignalCrossAttention(
                semantic_dim=embed_dim, physio_dim=108
            )

        # Shared projection layers (SwiGLU Reasoning Blocks)
        self.shared_proj = nn.Sequential(
            *[
                ExpansionBlock(
                    embed_dim, expansion_factor=expansion_factor, dropout=dropout
                )
                for _ in range(n_blocks)
            ],
            nn.LayerNorm(embed_dim),
        )

        self.head_2_1 = ClassificationHead(embed_dim, 1)
        self.head_2_2 = ClassificationHead(embed_dim, 1)
        self.head_2_3 = ClassificationHead(embed_dim, 5)

    def forward(self, ids: list[int], physio_features=None, physio_mask=None):
        """
        ids: List of sample IDs
        """
        # Map sample IDs to embedding indices
        indices = [self.id_to_embedding_idx[int(sample_id)] for sample_id in ids]
        indices = torch.tensor(indices, device=self.embeddings.device)

        # Retrieve raw Gemini embeddings
        features = self.embeddings[indices]

        # Inject Biosignals BEFORE SwiGLU
        if (
            self.use_sensorial
            and physio_features is not None
            and physio_mask is not None
        ):
            features = self.physio_fusion(features, physio_features, physio_mask)

        # Shared features (SwiGLU digests the text/image + human reaction)
        shared_features = self.shared_proj(features)

        # Output raw logits
        logits_2_1 = self.head_2_1(shared_features)

        # Soft gating: Dampen features for non-sexist memes before downstream tasks
        if self.soft_gating:
            prob_2_1 = torch.sigmoid(logits_2_1.detach())
            shared_features = shared_features * prob_2_1

        # Logits for Task 2.2 and 2.3
        logits_2_2 = self.head_2_2(shared_features)
        logits_2_3 = self.head_2_3(shared_features)

        return {
            "logits_2_1": logits_2_1,
            "logits_2_2": logits_2_2,
            "logits_2_3": logits_2_3,
        }
