import torch
import torch.nn as nn
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class Qwen(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = Qwen3VLEmbedder(model_name_or_path=model_name)

        embed_dim = 2048
        self.head_2_1 = ClassificationHead(embed_dim, 1)
        self.head_2_2 = ClassificationHead(embed_dim, 1)
        self.head_2_3 = ClassificationHead(embed_dim, 5)

        if freeze_backbone:
            for param in self.backbone.model.parameters():
                param.requires_grad = False

    def forward(self, image, text):
        # Get multimodal embeddings from Qwen3-VL-Embedding
        texts = [
            {"text": text[i]} for i in range(text.shape[0])
        ]
        imgs = [
            {"image": image[i]} for i in range(image.shape[0])
        ]
        inputs = texts + imgs
        embeddings = self.backbone.process(inputs)

        similarity_scores = embeddings[:text.shape[0]] @ embeddings[text.shape[0]:].T
        print(similarity_scores.tolist())

        # Output raw logits for BCEWithLogitsLoss
        logits_2_1 = self.head_2_1(embeddings[:text.shape[0]])
        logits_2_2 = self.head_2_2(embeddings[:text.shape[0]])
        logits_2_3 = self.head_2_3(embeddings[:text.shape[0]])

        return {
            "logits_2_1": logits_2_1,
            "logits_2_2": logits_2_2,
            "logits_2_3": logits_2_3,
        }
