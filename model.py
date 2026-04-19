import torch
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    MultilabelF1Score,
)
from models.Gemini import Gemini
from loss import CustomLoss


class EXISTModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "gemini",
        n_blocks: int = 2,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
    ):
        super().__init__()
        self.validation_step_outputs = []

        if model_name == "gemini":
            self.model = Gemini(
                n_blocks=n_blocks, expansion_factor=expansion_factor, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.criterion = CustomLoss()

        self.save_hyperparameters(ignore=["model"])

        self.metrics_2_1 = MetricCollection(
            {
                "f1": BinaryF1Score(threshold=0.5),
            },
            postfix="_2_1",
        )

        # Task 2.2: Highly subjective (Intention). Lower threshold to catch soft predictions.
        self.metrics_2_2 = MetricCollection(
            {
                "f1": BinaryF1Score(threshold=0.35),
            },
            postfix="_2_2",
        )

        # Task 2.3: Rare classes (Sexual Violence, etc.). Lower threshold.
        self.metrics_2_3 = MetricCollection(
            {
                "f1_macro": MultilabelF1Score(
                    num_labels=5, average="macro", threshold=0.3
                ),
            },
            postfix="_2_3",
        )

    def forward(
        self, image=None, text=None, ids=None, physio_features=None, physio_mask=None
    ):
        if isinstance(self.model, Gemini):
            return self.model(ids)
        if physio_features is None or physio_mask is None:
            return self.model(image, text)
        return self.model(image, text, physio_features, physio_mask)

    def _step(self, batch, batch_idx):
        image = batch.get("image", None)
        text = batch.get("text", None)
        ids = batch.get("id", None)
        physio_features = batch.get("physio_features", None)
        physio_mask = batch.get("physio_mask", None)

        if isinstance(self.model, Gemini):
            outputs = self.model(ids)
        elif physio_features is not None and physio_mask is not None:
            outputs = self.model(image, text, physio_features, physio_mask)
        else:
            outputs = self.model(image, text)

        targets = {
            "t_2_1": batch["target_2_1"],
            "t_2_2": batch["target_2_2"],
            "t_2_3": batch["target_2_3"],
        }

        masks = {
            "physio_mask": physio_mask,
            "cond_mask": (
                batch["target_2_1"] > 0
            ).float(),  # True if at least 1 annotator said it was sexist (target > 0)
        }

        return outputs, targets, masks

    def compute_metrics(self, outputs, targets, masks):
        # Convert logits to probabilities
        preds_2_1_prob = torch.sigmoid(outputs["logits_2_1"])
        preds_2_2_prob = torch.sigmoid(outputs["logits_2_2"])
        preds_2_3_prob = torch.sigmoid(outputs["logits_2_3"])

        t_2_1 = targets["t_2_1"]
        t_2_2 = targets["t_2_2"]
        t_2_3 = targets["t_2_3"]

        # --- Task 2.1 (Always Evaluated) ---
        hard_t_2_1 = (t_2_1 >= 0.5).int()

        # FIX: Update metrics INDIVIDUALLY by key to avoid broadcasting the wrong targets!
        self.metrics_2_1["f1"].update(preds_2_1_prob, hard_t_2_1)

        # --- Extract Conditional Mask (Only look at sexist memes) ---
        valid_mask = masks["cond_mask"].squeeze().bool()

        # Only evaluate 2.2 and 2.3 if there is at least one sexist meme in the batch
        if valid_mask.any():
            valid_preds_2_2 = preds_2_2_prob[valid_mask]
            valid_targets_2_2 = t_2_2[valid_mask]

            valid_preds_2_3 = preds_2_3_prob[valid_mask]
            valid_targets_2_3 = t_2_3[valid_mask]
            valid_preds_2_3 = valid_preds_2_3.reshape(-1, 5)
            valid_targets_2_3 = valid_targets_2_3.reshape(-1, 5)

            # --- Task 2.2 ---
            hard_t_2_2 = (valid_targets_2_2 >= 0.5).int()
            self.metrics_2_2["f1"].update(valid_preds_2_2, hard_t_2_2)

            # --- Task 2.3 ---
            hard_t_2_3 = (valid_targets_2_3 >= 0.5).int()
            self.metrics_2_3["f1_macro"].update(valid_preds_2_3, hard_t_2_3)

    def training_step(self, batch, batch_idx):
        outputs, targets, masks = self._step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets, masks)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        outputs, targets, masks = self._step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets, masks)
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.compute_metrics(outputs, targets, masks)

        step_output = {
            "loss": loss_dict["total_loss"],
            "id": batch["id"],
            "logits_2_1": outputs["logits_2_1"].detach(),
            "logits_2_2": outputs["logits_2_2"].detach(),
            "logits_2_3": outputs["logits_2_3"].detach(),
        }
        self.validation_step_outputs.append(step_output)
        return step_output

    def on_validation_epoch_end(self):
        # Log aggregated standard metrics and reset
        self.log_dict(
            {f"val/{k}": v for k, v in self.metrics_2_1.compute().items()},
            logger=True,
            prog_bar=False,
        )
        self.metrics_2_1.reset()

        try:
            self.log_dict(
                {f"val/{k}": v for k, v in self.metrics_2_2.compute().items()},
                logger=True,
                prog_bar=False,
            )
            self.log_dict(
                {f"val/{k}": v for k, v in self.metrics_2_3.compute().items()},
                logger=True,
                prog_bar=False,
            )
        except Exception:
            pass
        finally:
            self.metrics_2_2.reset()
            self.metrics_2_3.reset()

        # Clear step outputs to avoid memory bloat, since we're not using them for PyEvALL here
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        outputs, _, _ = self._step(batch, batch_idx)
        # Ensure we return floats for JSON serialization
        results = {
            "id": batch["id"],
            "logits_2_1": outputs["logits_2_1"].float(),
            "logits_2_2": outputs["logits_2_2"].float(),
            "logits_2_3": outputs["logits_2_3"].float(),
        }
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=total_steps,
            pct_start=self.warmup_ratio,
            max_lr=self.lr,
            anneal_strategy="cos",
        )

        print(f"Optimizer: {optimizer}")
        print(f"Scheduler: {scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
