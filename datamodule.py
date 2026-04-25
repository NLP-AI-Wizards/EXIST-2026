import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from PIL import Image
from typing import Optional
from functools import partial
from sklearn.model_selection import train_test_split


def collate_fn(batch, include_image=True, include_text=True, include_id=True):
    collated_batch = {}

    if include_id and "id" in batch[0]:
        collated_batch["id"] = [str(item["id"]) for item in batch]
    if include_image and "image" in batch[0]:
        collated_batch["image"] = [Image.fromarray(item["image"]) for item in batch]
    if include_text and "text" in batch[0]:
        collated_batch["text"] = [item["text"] for item in batch]

    target_2_1 = torch.stack([item["target_2_1"] for item in batch])
    target_2_2 = torch.stack([item["target_2_2"] for item in batch])
    target_2_3 = torch.stack([item["target_2_3"] for item in batch])

    collated_batch.update(
        {
            "target_2_1": target_2_1,
            "target_2_2": target_2_2,
            "target_2_3": target_2_3,
        }
    )

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
        train_dataset: torch.utils.data.Dataset = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        include_image: bool = True,
        include_text: bool = True,
        include_id: bool = True,
    ):
        super().__init__()
        self.train = train_dataset
        self.test = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.include_image = include_image
        self.include_text = include_text
        self.include_id = include_id

        self.collate = partial(
            collate_fn,
            include_image=self.include_image,
            include_text=self.include_text,
            include_id=self.include_id,
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            # Stratified split based on Task 2.1
            strata = []
            for _, row in self.train.data.iterrows():
                valid_2_1 = [l for l in row.get("labels_task2_1", []) if l in ["YES", "NO"]]
                t_2_1 = valid_2_1.count("YES") / len(valid_2_1) if len(valid_2_1) > 0 else 0.0
                strata.append(int(t_2_1 >= 0.5))

            train_idx, val_idx = train_test_split(
                range(len(self.train)),
                test_size=0.15,
                stratify=strata,
                random_state=self.seed
            )

            self.train_dataset = Subset(self.train, train_idx)
            self.val_dataset = Subset(self.train, val_idx)
            self.predict_dataset = self.train

        if self.test_dataset is None:
            self.test_dataset = self.test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def predict_dataloader(self):
        # Ensure setup is called if bypassing training
        if self.predict_dataset is None:
            self.setup(stage="predict")

        # Use full training dataset for predicting/evaluating against golds
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=False,
        )
