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


class AdvancedPhysioEncoder(nn.Module):
    def __init__(self, semantic_dim=768):
        super().__init__()

        # 1. ET and HR combined Encoder (24 + 4 = 28)
        self.et_hr_mlp = nn.Sequential(
            nn.Linear(28, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 128)
        )

        # 2. EEG 2D CNN Encoder (16 channels x 5 frequency bands)
        self.eeg_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((4, 1)),
            nn.Flatten(),  # Outputs 128 features
        )

        # 3. Final Fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(128 + 128, semantic_dim), nn.GELU(), nn.LayerNorm(semantic_dim)
        )

    def forward(self, et_features, hr_features, eeg_features):
        B, S, _ = et_features.shape

        # --- 1. Sanitize and Log1p SQUASH to fix huge reaction times ---
        def sanitize_and_squash(x):
            x = torch.nan_to_num(x, nan=0.0)
            return torch.sign(x) * torch.log1p(torch.abs(x))

        et_feat = sanitize_and_squash(et_features)
        hr_feat = sanitize_and_squash(hr_features)
        eeg_feat = sanitize_and_squash(eeg_features)

        # --- 2. Process ET/HR together ---
        et_hr_cat = torch.cat([et_feat, hr_feat], dim=-1)  # (B, S, 28)
        et_hr_embed = self.et_hr_mlp(et_hr_cat)  # (B, S, 128)

        # --- 3. Process EEG as a 2D Grid ---
        # Shape: (Batch * Subjects, 1 Channel, 16 Electrodes, 5 Frequencies)
        eeg_feat_2d = eeg_feat.view(B * S, 1, 16, 5)

        eeg_embed = self.eeg_cnn(eeg_feat_2d)  # (B*S, 128)
        eeg_embed = eeg_embed.view(B, S, 128)  # (B, S, 128)

        # --- 4. Fuse representations ---
        combined_physio = torch.cat([et_hr_embed, eeg_embed], dim=-1)  # (B, S, 256)
        V_physio = self.fusion_proj(combined_physio)  # (B, S, 768)

        return V_physio


class BiosignalCrossAttention(nn.Module):
    def __init__(self, semantic_dim=768):
        super().__init__()

        self.physio_encoder = AdvancedPhysioEncoder(semantic_dim=semantic_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=semantic_dim // 64,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(semantic_dim)

        self.out_proj = nn.Linear(semantic_dim, semantic_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, V_semantic, et_features, hr_features, eeg_features, physio_mask):

        # Process the raw 108D vector using the CNN + MLP split
        V_physio = self.physio_encoder(et_features, hr_features, eeg_features)

        query = V_semantic.unsqueeze(1)
        attn_padding_mask = ~physio_mask.bool()

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
        attended_physio[all_masked] = 0.0

        physio_update = torch.tanh(self.out_proj(self.norm(attended_physio)))
        return V_semantic + physio_update


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
            self.physio_fusion = BiosignalCrossAttention(semantic_dim=embed_dim)

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
        indices = [self.id_to_embedding_idx[int(sample_id)] for sample_id in ids]
        indices = torch.tensor(indices, device=self.embeddings.device)

        # Pure Semantic Features (Gemini Embedding 2)
        semantic_features = self.embeddings[indices]

        # Pure Semantics through the SwiGLU blocks (Tasks 2.2 and 2.3)
        pure_shared_features = self.shared_proj(semantic_features)

        # Enrich semantic featues with physio reactions (Task 2.1)
        if (
            self.use_sensorial
            and physio_features is not None
            and physio_mask is not None
        ):
            physio_reaction_features = self.physio_fusion(
                semantic_features,
                physio_features[0],  # et
                physio_features[1],  # hr
                physio_features[2],  # eeg
            )
        else:
            physio_reaction_features = semantic_features

        # Process Physio Reaction through SwiGLU block
        physio_shared_features = self.shared_proj(physio_reaction_features)

        # Head 2.1 (Is it sexist?) gets the Biosignal + Semantic features
        logits_2_1 = self.head_2_1(physio_shared_features)

        # soft-gating from 2.1 probs
        if self.soft_gating:
            prob_2_1 = torch.sigmoid(logits_2_1.detach())
            physio_shared_features = physio_shared_features * prob_2_1
            pure_shared_features = pure_shared_features * prob_2_1

        # Head 2.2: Biosignal + Semantic features (optional soft-gating)
        logits_2_2 = self.head_2_2(physio_shared_features)

        # Head 2.3: Pure Semantic features only (optional soft-gating)
        logits_2_3 = self.head_2_3(pure_shared_features)

        return {
            "logits_2_1": logits_2_1,
            "logits_2_2": logits_2_2,
            "logits_2_3": logits_2_3,
        }
