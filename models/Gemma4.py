import torch
import torch.nn as nn
from transformers import AutoModelForMultimodalLM, AutoProcessor


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class Gemma4(nn.Module):
    def __init__(
        self,
        model_name: str = "google/gemma-4-E2B-it",
        freeze_backbone: bool = True,
        max_length: int = 2048,
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_length = max_length

        embed_dim = self.model.config.text_config.hidden_size
        self.head_2_1 = ClassificationHead(embed_dim, 1)
        self.head_2_2 = ClassificationHead(embed_dim, 1)
        self.head_2_3 = ClassificationHead(embed_dim, 5)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _build_messages(self, image, text):
        system_prompt = (
            "You are an Artificial Intelligence for sexism detection and "
            "classification in social media contents."
        )

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]

    def _pool_last_token(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        last_token_idx = attention_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[batch_idx, last_token_idx]

    def forward(self, image, text):
        conversations = [
            self._build_messages(image[i], text[i]) for i in range(len(text))
        ]

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_state = outputs.hidden_states[-1]
        pooled = self._pool_last_token(hidden_state, inputs["attention_mask"])

        self.head_2_1.to(pooled.device)
        self.head_2_2.to(pooled.device)
        self.head_2_3.to(pooled.device)

        logits_2_1 = self.head_2_1(pooled)
        logits_2_2 = self.head_2_2(pooled)
        logits_2_3 = self.head_2_3(pooled)

        if logits_2_1.ndim == 3 and logits_2_1.shape[1] == 1:
            logits_2_1 = logits_2_1.squeeze(1)
        if logits_2_2.ndim == 3 and logits_2_2.shape[1] == 1:
            logits_2_2 = logits_2_2.squeeze(1)
        if logits_2_3.ndim == 3 and logits_2_3.shape[1] == 1:
            logits_2_3 = logits_2_3.squeeze(1)

        return {
            "logits_2_1": logits_2_1,
            "logits_2_2": logits_2_2,
            "logits_2_3": logits_2_3,
        }
