import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image
from typing import Optional


def collate_fn(batch):
    ids = [str(item["id"]) for item in batch]
    images = [Image.fromarray(item["image"]) for item in batch]
    texts = [item["text"] for item in batch]

    target_2_1 = torch.stack([item["target_2_1"] for item in batch])
    target_2_2 = torch.stack([item["target_2_2"] for item in batch])
    target_2_3 = torch.stack([item["target_2_3"] for item in batch])

    collated_batch = {
        "id": ids,
        "image": images,
        "text": texts,
        "target_2_1": target_2_1,
        "target_2_2": target_2_2,
        "target_2_3": target_2_3,
    }

    if "physio_features" in batch[0]:
        collated_batch["physio_features"] = torch.stack(
            [item["physio_features"] for item in batch]
        )
        collated_batch["physio_mask"] = torch.stack(
            [item["physio_mask"] for item in batch]
        )

    return collated_batch


class EXISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.train = train_dataset
        self.test = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            # Use full training dataset for training
            self.train_dataset = self.train

        if self.val_dataset is None:
            # Use full test dataset for validation
            self.val_dataset = self.test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        # Ensure setup is called if bypassing training
        if self.train_dataset is None:
            self.setup(stage="predict")

        # Use full training dataset for predicting/evaluating against golds
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
