import torch
import torch.nn as nn

from transformers import AutoProcessor, AutoModel

class PhysioEncoder(nn.Module):
    def __init__(self, physio_dim: int = 108, hidden_dim: int = 256):
        super().__init__()
        # TODO: CNN for EGG and MLP for ET and HRV (?)
        self.encoder = nn.Sequential(
            nn.Linear(physio_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class DISCERN(nn.Module):
    """
    Disagreement Identification via Sensorial Cross-attention and Enriched Representation Networks.
    """
    def __init__(
        self,
        physio_dim: int = 108,
        hidden_dim: int = 512,
        num_classes_2_3: int = 5,
        activation: nn.Module = nn.GELU(),
        num_heads: int = 8,
        dropout: float = 0.1,
        backbone_name: str = "google/siglip-base-patch16-224",
        freeze_backbone: bool = True,
        adapt_backbone: bool = False,
    ):
        super().__init__()

        # HateCLIP/SigLIP backbone for multimodal feature extraction
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.processor = AutoProcessor.from_pretrained(backbone_name)

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if adapt_backbone:
            # Unfreeze the LAST 2 layers of the Vision Encoder
            for layer in self.backbone.vision_model.encoder.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Unfreeze the LAST 2 layers of the Text Encoder
            for layer in self.backbone.text_model.encoder.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Unfreeze the final projection heads
            for param in self.backbone.vision_model.head.parameters():
                param.requires_grad = True
            for param in self.backbone.text_model.head.parameters():
                param.requires_grad = True

        # SigLIP-base always outputs 768. If our hidden_dim is different, we must project it.
        siglip_out_dim = 768
        self.siglip_proj = nn.Linear(siglip_out_dim, hidden_dim) if siglip_out_dim != hidden_dim else nn.Identity()

        # Physio Encoder
        self.physio_encoder = PhysioEncoder(physio_dim, hidden_dim)

        # Cross-Attention Module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Multimodal Fusion Projection
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Dropout(dropout)
        )

        # Hierarchical MTL Classification Heads
        self.head_2_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Linear(hidden_dim // 2, 1),
        )

        self.head_2_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Linear(hidden_dim // 2, 1),
        )

        self.head_2_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Linear(hidden_dim // 2, num_classes_2_3),
        )

    def forward(self, image, text, physio_features, physio_mask):
        device = physio_features.device

        physio_features = torch.nan_to_num(
            physio_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        ## Process text
        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).to(device)

        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        clip_outputs = self.backbone(**text_inputs, pixel_values=image.to(device))

        image_embeds = clip_outputs.image_embeds
        text_embeds = clip_outputs.text_embeds

        # Combine Text and Image
        V_stimulus = 0.5 * (image_embeds + text_embeds)

        # ADDED: Map SigLIP's 768 output to the network's configured hidden_dim
        V_stimulus = self.siglip_proj(V_stimulus)

        # Encode Physiological Features
        V_physio = self.physio_encoder(physio_features)

        # Cross-Attention Fusion
        query = V_stimulus.unsqueeze(1)
        attn_padding_mask = ~physio_mask.bool().to(device)

        # Find memes that have NO physiological data (all elements are True)
        all_masked = attn_padding_mask.all(dim=-1)

        # Temporarily unmask the first token for these empty memes to prevent NaN
        # (It will just attend to the zero-padded vector, which we will remove anyway)
        attn_padding_mask[all_masked, 0] = False

        attended_physio, _ = self.cross_attn(
            query=query,
            key=V_physio,
            value=V_physio,
            key_padding_mask=attn_padding_mask,
        )

        attended_physio = attended_physio.squeeze(1)

        # Force the attended output to be exactly 0.0 for memes without physio data
        attended_physio[all_masked] = 0.0

        # Concat and Final Multimodal Fusion
        V_fused = torch.cat([V_stimulus, attended_physio], dim=-1)
        V_fused = self.fusion_layer(V_fused)

        # Multi-Task Predictions
        p_2_1 = torch.sigmoid(self.head_2_1(V_fused))
        p_2_2 = torch.sigmoid(self.head_2_2(V_fused))
        p_2_3 = torch.sigmoid(self.head_2_3(V_fused))

        return {
            "preds_2_1": p_2_1,
            "preds_2_2": p_2_2,
            "preds_2_3": p_2_3,
            "V_stimulus": V_stimulus,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "V_physio": V_physio,
        }

def discern_tiny(physio_dim=108, hidden_dim=256):
    return DISCERN(
        physio_dim=physio_dim,
        hidden_dim=hidden_dim,
        activation=nn.GELU(),
        num_heads=4,
        dropout=0.1
    )

def discern_base(physio_dim=108, hidden_dim=512):
    return DISCERN(
        physio_dim=physio_dim,
        hidden_dim=hidden_dim,
        activation=nn.GELU(),
        num_heads=8,
        dropout=0.1
    )

def discern_large(physio_dim=108, hidden_dim=768):
    return DISCERN(
        physio_dim=physio_dim,
        hidden_dim=hidden_dim,
        activation=nn.GELU(),
        num_heads=12,
        dropout=0.1
    )