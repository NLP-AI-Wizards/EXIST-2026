import torch
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    MultilabelF1Score,
)
from torchmetrics.regression import MeanAbsoluteError

from models.DISCERN import discern_tiny, discern_base, discern_large
from models.SigLIP import SigLIP
from models.Qwen import Qwen
from loss import CustomLoss

class EXISTModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "discern_tiny",
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        super().__init__()

        if model_name == "discern_tiny":
            self.model = discern_tiny()
        elif model_name == "discern_base":
            self.model = discern_base()
        elif model_name == "discern_large":
            self.model = discern_large()
        elif model_name == "siglip":
            self.model = SigLIP()
        elif model_name == "qwen":
            self.model = Qwen()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.criterion = CustomLoss()

        self.save_hyperparameters(ignore=["model"])

        # EXIST TRACKS:
        # F1 Score handles the "Hard" Majority-Vote Track
        # MAE handles the "Soft" Probabilistic Track

        self.metrics_2_1 = MetricCollection(
            {
                "f1": BinaryF1Score(),
                "mae": MeanAbsoluteError(),
            },
            postfix="_2_1",
        )

        self.metrics_2_2 = MetricCollection(
            {
                "f1": BinaryF1Score(),
                "mae": MeanAbsoluteError(),
            },
            postfix="_2_2",
        )

        self.metrics_2_3 = MetricCollection(
            {
                "f1_macro": MultilabelF1Score(num_labels=5, average="macro"),
                "mae": MeanAbsoluteError(),
            },
            postfix="_2_3",
        )

    def forward(self, image, text, physio_features=None, physio_mask=None):
        if physio_features is None or physio_mask is None:
            return self.model(image, text)
        return self.model(image, text, physio_features, physio_mask)

    def _step(self, batch, batch_idx):
        image = batch["image"]
        text = batch["text"]
        physio_features = batch.get("physio_features", None)
        physio_mask = batch.get("physio_mask", None)

        if physio_features is not None and physio_mask is not None:
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
        # ----------------------------------------
        # METRIC UPDATES
        # ----------------------------------------

        # Convert logits to probabilities for metric calculation
        preds_2_1_prob = torch.sigmoid(outputs["logits_2_1"])
        preds_2_2_prob = torch.sigmoid(outputs["logits_2_2"])
        preds_2_3_prob = torch.sigmoid(outputs["logits_2_3"])

        t_2_1 = targets["t_2_1"]
        t_2_2 = targets["t_2_2"]
        t_2_3 = targets["t_2_3"]

        # --- Task 2.1 (Always Evaluated) ---
        hard_t_2_1 = (t_2_1 >= 0.5).int()  # Majority vote threshold
        self.metrics_2_1.update(preds_2_1_prob, hard_t_2_1)
        # Update MAE directly with soft probabilities
        self.metrics_2_1["mae_2_1"].update(preds_2_1_prob, t_2_1)

        # --- Extract Conditional Mask (Only look at sexist memes) ---
        # Squeeze to 1D boolean array
        valid_mask = masks["cond_mask"].squeeze().bool()

        # Only evaluate 2.2 and 2.3 if there is at least one sexist meme in the batch
        if valid_mask.any():
            # Filter the tensors
            valid_preds_2_2 = preds_2_2_prob[valid_mask]
            valid_targets_2_2 = t_2_2[valid_mask]

            valid_preds_2_3 = preds_2_3_prob[valid_mask]
            valid_targets_2_3 = t_2_3[valid_mask]

            # --- Task 2.2 ---
            hard_t_2_2 = (valid_targets_2_2 >= 0.5).int()
            self.metrics_2_2.update(valid_preds_2_2, hard_t_2_2)
            self.metrics_2_2["mae_2_2"].update(valid_preds_2_2, valid_targets_2_2)

            # --- Task 2.3 ---
            hard_t_2_3 = (valid_targets_2_3 >= 0.5).int()
            self.metrics_2_3.update(valid_preds_2_3, hard_t_2_3)
            self.metrics_2_3["mae_2_3"].update(valid_preds_2_3, valid_targets_2_3)

    def training_step(self, batch, batch_idx):
        outputs, targets, masks = self._step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets, masks)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
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
            logger=True
        )
        self.compute_metrics(outputs, targets, masks)

        return loss_dict["total_loss"]

    def test_step(self, batch, batch_idx):
        outputs, targets, masks = self._step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets, masks)
        self.log_dict(
            {f"test/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.compute_metrics(outputs, targets, masks)
        self.log_dict(self.metrics_2_1, on_step=False, on_epoch=True)
        self.log_dict(self.metrics_2_2, on_step=False, on_epoch=True)
        self.log_dict(self.metrics_2_3, on_step=False, on_epoch=True)

        return loss_dict["total_loss"]

    def predict_step(self, batch, batch_idx):
        outputs, _, _ = self._step(batch, batch_idx)
        return outputs

    def on_validation_epoch_end(self):
        self.log_dict({f"val/{k}": v for k, v in self.metrics_2_1.compute().items()})
        self.metrics_2_1.reset()

        # Handle tasks 2.2 and 2.3 safely (in case no sexist memes were in the val set)
        try:
            self.log_dict(
                {f"val/{k}": v for k, v in self.metrics_2_2.compute().items()}
            )
            self.log_dict(
                {f"val/{k}": v for k, v in self.metrics_2_3.compute().items()}
            )
        except Exception:
            pass
        finally:
            self.metrics_2_2.reset()
            self.metrics_2_3.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.1,
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
                "monitor": "val/total_loss",
            },
        }