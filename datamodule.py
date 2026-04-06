import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Optional


def collate_fn(batch):
    images = [Image.fromarray(item["image"]) for item in batch]
    texts = [item["text"] for item in batch]

    target_2_1 = torch.stack([item["target_2_1"] for item in batch])
    target_2_2 = torch.stack([item["target_2_2"] for item in batch])
    target_2_3 = torch.stack([item["target_2_3"] for item in batch])

    collated_batch = {
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
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        n_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.n_samples = n_samples

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            # Get all indices
            indices = np.arange(len(self.dataset))
            if self.n_samples is not None:
                indices = np.random.choice(indices, size=self.n_samples, replace=False)

            # Extract labels for stratification from the underlying DataFrame
            # We binarize Task 2.1 marginals to 0 or 1 for the split logic
            raw_labels = self.dataset.data.iloc[indices]["labels_task2_1"]
            stratify_labels = raw_labels.apply(
                lambda x: 1 if (x.count("YES") / len(x)) >= 0.5 else 0
            ).values

            # Split 70% Train | 30% Temp (Val + Test)
            train_idx, temp_idx, _, y_temp = train_test_split(
                indices,
                stratify_labels,
                test_size=0.30,  # 30% for Val + Test
                random_state=self.seed,
                stratify=stratify_labels,
            )

            # Split 30% Temp into 50/50 (15% Val | 15% Test)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,  # Half of 30% is 15%
                random_state=self.seed,
                stratify=y_temp,  # Stratify based on the remaining labels
            )

            # Create Subsets
            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
            self.test_dataset = Subset(self.dataset, test_idx)

            print(f"📊 Dataset Split Complete (Seed {self.seed}):")
            print(
                f"   - Train: {len(self.train_dataset)} ({len(self.train_dataset) / len(self.dataset):.1%})"
            )
            print(
                f"   - Val:   {len(self.val_dataset)} ({len(self.val_dataset) / len(self.dataset):.1%})"
            )
            print(
                f"   - Test:  {len(self.test_dataset)} ({len(self.test_dataset) / len(self.dataset):.1%})"
            )

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
