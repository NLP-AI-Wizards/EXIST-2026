import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataset import EXISTDataset

def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    target_2_1 = torch.stack([item['target_2_1'] for item in batch])
    target_2_2 = torch.stack([item['target_2_2'] for item in batch])
    target_2_3 = torch.stack([item['target_2_3'] for item in batch])

    collated_batch = {
        "image": images,
        "text": texts,
        "target_2_1": target_2_1,
        "target_2_2": target_2_2,
        "target_2_3": target_2_3,
    }

    if 'physio_features' in batch[0]:
        collated_batch['physio_features'] = torch.stack([item['physio_features'] for item in batch])
        collated_batch['physio_mask'] = torch.stack([item['physio_mask'] for item in batch])

    return collated_batch

class EXISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: EXISTDataset,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Split dataset into train and validation (80/20 split)
            total_size = len(self.dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        if stage == "test" or stage is None:
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
