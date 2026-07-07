import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipModel

from models.model_head.mlp import ClassificationHead, SwiGLU


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


class SigLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        n_blocks: int = 2,
        expansion_factor: int = 4,
        dropout: float = 0.2,
        soft_gating: bool = True,
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.siglip = SiglipModel.from_pretrained(model_name)
        self.soft_gating = soft_gating

        embed_dim = 768 * 2

        # Shared projection layers
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

        for param in self.siglip.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, text: list[str]):
        """
        image: Tensor of shape (Batch, H, W, C) from the dataloader
        text: List of strings
        """
        device = next(self.parameters()).device

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        # Move processed tokens/pixels to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Pass through SigLIP backbone
        outputs = self.siglip(**inputs)

        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

        # Concatenate modalities
        # Shape: (Batch, 1536)
        combined = torch.cat([text_embeds, image_embeds], dim=1)

        shared_features = self.shared_proj(combined)

        # Output raw logits for BCEWithLogitsLoss
        logits_2_1 = self.head_2_1(shared_features)

        # Added block, soft gating
        if self.soft_gating:
            prob_2_1 = torch.sigmoid(logits_2_1.detach())
            shared_features = shared_features * prob_2_1

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
