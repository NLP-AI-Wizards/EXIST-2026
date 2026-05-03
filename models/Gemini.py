import numpy as np
import torch
import torch.nn as nn
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


class SubjectFiLM(nn.Module):
    def __init__(self, subject_embed_dim=32, feature_dim=256):
        super().__init__()
        # Takes the subject embedding and generates Scale (Gamma) and Shift (Beta)
        self.film_generator = nn.Linear(subject_embed_dim, feature_dim * 2)

    def forward(self, features, subject_embeds):
        """
        features: (Batch, Subjects, Feature_Dim) - The concatenated EEG/ET/HR data
        subject_embeds: (Batch, Subjects, Embed_Dim) - The dense ID vectors
        """
        film_params = self.film_generator(subject_embeds)

        # Split into scale and shift
        gamma, beta = film_params.chunk(2, dim=-1)

        # Apply FiLM conditioning
        # We add 1.0 to gamma so the default scaling is identity (1 * x + 0)
        return features * (1.0 + gamma) + beta


class AdvancedPhysioEncoder(nn.Module):
    def __init__(self, semantic_dim=768, use_subject_ids=False, num_unique_subjects=13):
        # ID Subject: EN[1-7] + ES[1-5,8]
        super().__init__()
        self.use_subject_ids = use_subject_ids

        if use_subject_ids:
            # Subject ID Embedding Layer
            # padding_idx=0 ensures that padded/missing subjects output a vector of pure 0.0s
            subject_embed_dim = 32
            self.subject_embedding = nn.Embedding(
                num_embeddings=num_unique_subjects + 1,
                embedding_dim=subject_embed_dim,
                padding_idx=0,
            )

        # ET and HR combined Encoder (24 + 4 = 28)
        self.et_hr_mlp = nn.Sequential(nn.Linear(28, 128), nn.GELU(), nn.LayerNorm(128))

        # EEG 2D CNN Encoder (16 channels x 5 frequency bands)
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

        if use_subject_ids:
            # Subject Conditioning (FiLM)
            # Modulates the 256 concatenated features (128 ET/HR + 128 EEG)
            self.film_layer = SubjectFiLM(
                subject_embed_dim=subject_embed_dim, feature_dim=256
            )

        # Final Fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(256, semantic_dim), nn.GELU(), nn.LayerNorm(semantic_dim)
        )

    def forward(self, id_features, et_features, hr_features, eeg_features):
        B, S, _ = et_features.shape

        if self.use_subject_ids:
            # Map IDs to dense subject profiles
            # Shape: (B, S, 32)
            subject_embeds = self.subject_embedding(id_features)

        # Sanitize and Squash continuous sensors
        def sanitize_and_squash(x):
            x = torch.nan_to_num(x, nan=0.0)
            return torch.sign(x) * torch.log1p(torch.abs(x))

        et_feat = sanitize_and_squash(et_features)
        hr_feat = sanitize_and_squash(hr_features)
        eeg_feat = sanitize_and_squash(eeg_features)

        # Process ET/HR
        et_hr_cat = torch.cat([et_feat, hr_feat], dim=-1)  # (B, S, 28)
        et_hr_embed = self.et_hr_mlp(et_hr_cat)  # (B, S, 128)

        # Process EEG
        eeg_feat_2d = eeg_feat.view(B * S, 1, 16, 5)
        eeg_embed = self.eeg_cnn(eeg_feat_2d)  # (B*S, 128)
        eeg_embed = eeg_embed.view(B, S, 128)  # (B, S, 128)

        # Combine Physiologies
        combined_physio = torch.cat([et_hr_embed, eeg_embed], dim=-1)  # (B, S, 256)

        if self.use_subject_ids:
            # FiLM Conditioning: Modulate the combined physio features based on subject profile
            conditioned_physio = self.film_layer(combined_physio, subject_embeds)

        # Final Fusion to match semantic dimension
        V_physio = self.fusion_proj(conditioned_physio)  # (B, S, 768)

        return V_physio


class BiosignalCrossAttention(nn.Module):
    def __init__(self, semantic_dim=768, use_subject_ids=False):
        super().__init__()

        self.physio_encoder = AdvancedPhysioEncoder(
            semantic_dim=semantic_dim, use_subject_ids=use_subject_ids
        )

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
        use_subject_ids: bool = False,
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
                semantic_dim=embed_dim, use_subject_ids=use_subject_ids
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
