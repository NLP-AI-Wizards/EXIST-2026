from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipModel


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SigLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.siglip = SiglipModel.from_pretrained(
            model_name,
            device_map="auto",
            dtype="bfloat16" if torch.cuda.is_available() else torch.float32,
        )
        embed_dim = 768
        self.head_2_1 = ClassificationHead(embed_dim * 2, 1)
        self.head_2_2 = ClassificationHead(embed_dim * 2, 1)
        self.head_2_3 = ClassificationHead(embed_dim * 2, 5)

        if freeze_backbone:
            for param in self.siglip.parameters():
                param.requires_grad = False

    def forward(self, image: np.ndarray, text: list[str]):
        device = next(self.parameters()).device

        inputs = self.processor(
            text=text,
            images=[Image.fromarray(image[i]) for i in range(len(image))],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.siglip(**inputs)

        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

        combined = torch.cat([text_embeds, image_embeds], dim=1)

        # Output raw logits for BCEWithLogitsLoss
        logits_2_1 = self.head_2_1(combined)
        logits_2_2 = self.head_2_2(combined)
        logits_2_3 = self.head_2_3(combined)

        return {
            "logits_2_1": logits_2_1,
            "logits_2_2": logits_2_2,
            "logits_2_3": logits_2_3,
        }
