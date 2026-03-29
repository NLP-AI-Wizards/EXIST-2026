import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import EXISTDataset

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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    from torchvision import transforms

    # Define standard image transformation for testing
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Instantiate dataset with ALL modalities toggled ON
    dataset = EXISTDataset(
        json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
        img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
        use_sensorial=True,
        use_annotator_metadata=True,
        transform=transform,
    )

    data_module = EXISTDataModule(dataset=dataset, batch_size=4, num_workers=0)
    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
