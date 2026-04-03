import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipModel, SiglipConfig


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SigLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-256-multilingual",
        max_length: int = 128,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name)
        config = SiglipConfig.from_pretrained(model_name)
        config.text_config.max_position_embeddings = max_length

        self.siglip = SiglipModel.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        self._resize_position_embeddings(max_length)

        embed_dim = 768

        self.head_2_1 = ClassificationHead(embed_dim * 2, 1)
        self.head_2_2 = ClassificationHead(embed_dim * 2, 1)
        self.head_2_3 = ClassificationHead(embed_dim * 2, 5)

        if freeze_backbone:
            for param in self.siglip.parameters():
                param.requires_grad = False

    def _resize_position_embeddings(self, new_max_len: int):
        """Resize text positional embeddings safely"""
        old_embed = self.siglip.text_model.embeddings.position_embedding
        old_len, dim = old_embed.weight.shape

        if new_max_len <= old_len:
            return  # nothing to do

        new_embed = nn.Embedding(new_max_len, dim)

        # copy pretrained weights
        new_embed.weight.data[:old_len] = old_embed.weight.data

        # initialize extra positions (mean is more stable than repeat)
        mean_vec = old_embed.weight.data.mean(dim=0, keepdim=True)
        new_embed.weight.data[old_len:] = mean_vec.repeat(new_max_len - old_len, 1)

        self.siglip.text_model.embeddings.position_embedding = new_embed

    def forward(self, image, text):
        device = next(self.parameters()).device

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.siglip(**inputs, interpolate_pos_encoding=True)

        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

        combined = torch.cat([text_embeds, image_embeds], dim=1)

        preds_2_1 = torch.sigmoid(self.head_2_1(combined))
        preds_2_2 = torch.sigmoid(self.head_2_2(combined))
        preds_2_3 = self.head_2_3(combined)

        return {
            "preds_2_1": preds_2_1,
            "preds_2_2": preds_2_2,
            "preds_2_3": preds_2_3,
        }
