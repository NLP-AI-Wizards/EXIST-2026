from functools import partial
from typing import Optional

import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


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

    collated_batch.update({
        "target_2_1": target_2_1,
        "target_2_2": target_2_2,
        "target_2_3": target_2_3,
    })

    if "physio_mask" in batch[0]:
        collated_batch["et_features"] = torch.stack([
            item["et_features"] for item in batch
        ])
        collated_batch["hr_features"] = torch.stack([
            item["hr_features"] for item in batch
        ])
        collated_batch["eeg_features"] = torch.stack([
            item["eeg_features"] for item in batch
        ])
        collated_batch["physio_mask"] = torch.stack([
            item["physio_mask"] for item in batch
        ])

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
        self.eval = test_dataset
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
        # Predict-only mode: if an external evaluation dataset was provided,
        # use it directly without creating train/val/test splits.
        if stage == "predict":
            if self.eval is not None:
                self.predict_dataset = self.eval
                return

        if self.train_dataset is None:
            # Stratified split based on Task 2.1
            strata = []
            for _, row in self.train.data.iterrows():
                valid_2_1 = [
                    l for l in row.get("labels_task2_1", []) if l in ["YES", "NO"]
                ]
                t_2_1 = (
                    valid_2_1.count("YES") / len(valid_2_1)
                    if len(valid_2_1) > 0
                    else 0.0
                )
                strata.append(int(t_2_1 >= 0.5))

            all_idx = list(range(len(self.train)))

            train_idx, temp_idx = train_test_split(
                all_idx,
                test_size=0.20,
                stratify=strata,
                random_state=self.seed,
            )

            temp_strata = [strata[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,
                stratify=temp_strata,
                random_state=self.seed,
            )

            self.train_dataset = Subset(self.train, train_idx)
            self.val_dataset = Subset(self.train, val_idx)
            self.test_dataset = Subset(self.train, test_idx)
            self.predict_dataset = self.test_dataset

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

        # Use internal test split for prediction.
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=False,
        )
