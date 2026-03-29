import torch
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    MultilabelAccuracy,
    MultilabelF1Score
)
from torchmetrics.regression import MeanAbsoluteError

from architecture import discern_tiny, discern_base, discern_large
from loss import DISCERNLoss

class EXISTModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "discern_tiny",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.98),
    ):
        super().__init__()

        if model_name == "discern_tiny":
            self.model = discern_tiny()
        elif model_name == "discern_base":
            self.model = discern_base()
        elif model_name == "discern_large":
            self.model = discern_large()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.criterion = DISCERNLoss()

        self.save_hyperparameters(ignore=["model"])

        self.metrics_2_1 = MetricCollection(
            {
                "acc": BinaryAccuracy(threshold=0.5),
                "f1": BinaryF1Score(threshold=0.5),
                "mae": MeanAbsoluteError(),
            },
            postfix="_2_1",
        )

        # Task 2.2: Source Intention (Conditional Binary)
        self.metrics_2_2 = MetricCollection(
            {
                "acc": BinaryAccuracy(threshold=0.5),
                "f1": BinaryF1Score(threshold=0.5),
                "mae": MeanAbsoluteError(),
            },
            postfix="_2_2",
        )

        # Task 2.3: Categorization (Conditional Multi-Label, 5 classes)
        self.metrics_2_3 = MetricCollection(
            {
                "acc": MultilabelAccuracy(num_labels=5, threshold=0.5),
                "f1_macro": MultilabelF1Score(
                    num_labels=5, threshold=0.5, average="macro"
                ),
                "mae": MeanAbsoluteError(),
            },
            postfix="_2_3",
        )

    def forward(self, image, text, physio=None):
        return self.model(image, text, physio)

    def _step(self, batch, batch_idx):
        image = batch["image"]
        text = batch["text"]
        physio_features = batch["physio_features"]

        outputs = self.model(
            image=image, text=text, physio_features=physio_features, physio_mask=batch["physio_mask"]
        )

        targets = {
            "t_2_1": batch["target_2_1"],
            "t_2_2": batch["target_2_2"],
            "t_2_3": batch["target_2_3"],
        }

        masks = {
            "cond_mask": batch["mask_conditional"],
            "physio_mask": batch["physio_mask"],
        }

        return outputs, targets, masks

    def compute_metrics(self, outputs, targets, masks):
        # ----------------------------------------
        # METRIC UPDATES
        # ----------------------------------------

        # Update Task 2.1 (Always computed)
        # Convert soft targets > 0.5 to hard targets (1 or 0) for classification metrics
        hard_t_2_1 = (targets["t_2_1"] > 0.5).int()

        self.metrics_2_1.update(
            preds=outputs["preds_2_1"],
            target=hard_t_2_1,
            # For MAE, we need to pass the raw probabilities to compare soft fit
            mae_kwargs={"preds": outputs["preds_2_1"], "target": targets["t_2_1"]},
        )

        # Extract conditional mask (True where meme is sexist)
        # Shape is (Batch, 1), we squeeze it to get a 1D boolean mask
        valid_mask = masks["cond_mask"].squeeze() > 0

        # Only evaluate 2.2 and 2.3 if there is at least one sexist meme in the batch
        if valid_mask.any():
            # Filter predictions and targets using the mask
            valid_preds_2_2 = outputs["preds_2_2"][valid_mask]
            valid_targets_2_2 = targets["t_2_2"][valid_mask]

            valid_preds_2_3 = outputs["preds_2_3"][valid_mask]
            valid_targets_2_3 = targets["t_2_3"][valid_mask]

            # Update Task 2.2
            hard_t_2_2 = (valid_targets_2_2 > 0.5).int()
            self.metrics_2_2.update(
                preds=valid_preds_2_2,
                target=hard_t_2_2,
                mae_kwargs={"preds": valid_preds_2_2, "target": valid_targets_2_2},
            )

            # Update Task 2.3
            hard_t_2_3 = (valid_targets_2_3 > 0.5).int()
            self.metrics_2_3.update(
                preds=valid_preds_2_3,
                target=hard_t_2_3,
                mae_kwargs={"preds": valid_preds_2_3, "target": valid_targets_2_3},
            )

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
        self.log(
            "test/loss",
            loss_dict["total_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.compute_metrics(outputs, targets, masks)

        return loss_dict["total_loss"]

    def predict_step(self, batch, batch_idx):
        outputs, _, _ = self._step(batch, batch_idx)
        return outputs

    def on_validation_epoch_end(self):
        # Compute and log the metrics at the end of the epoch
        metrics_dict = {}

        metrics_dict.update(self.metrics_2_1.compute())
        self.metrics_2_1.reset()

        # Safely compute 2.2 and 2.3 (in case the entire validation set had no sexist memes, which is unlikely)
        try:
            metrics_dict.update(self.metrics_2_2.compute())
            self.metrics_2_2.reset()

            metrics_dict.update(self.metrics_2_3.compute())
            self.metrics_2_3.reset()
        except ValueError:
            pass  # No valid samples in this epoch

        # Log all metrics with a "val_" prefix
        self.log_dict(
            {f"val/{k}": v for k, v in metrics_dict.items()}, prog_bar=True, logger=True
        )

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
            total_steps=self.trainer.estimated_stepping_batches,
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