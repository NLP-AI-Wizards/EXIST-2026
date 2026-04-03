import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional

TASK_2_3_CLASSES = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]


class EXISTDataset(Dataset):
    def __init__(
        self,
        json_path: str = r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
        img_dir: str = r"data/EXIST 2026 Memes Dataset/training/images",
        use_sensorial: bool = False,
        use_annotator_metadata: bool = False,
        transform=None,
        max_subjects: int = 4,
        n_samples: Optional[int] = None,
        physio_dim: int = 108,
    ):
        """
        Args:
            json_path (str): Path to the EXIST2026 JSON file.
            img_dir (str): Base directory containing the meme images.
            use_sensorial (bool): If True, extracts and pads physiological data.
            use_annotator_metadata (bool): If True, returns annotator demographics.
            transform (callable, optional): Torchvision transforms or CLIP processor.
            max_subjects (int): Maximum number of physiological subjects to pad to (4 per guidelines).
            n_samples (int, optional): Number of samples to use from the dataset. If None, uses the entire dataset.
            physio_dim (int): Total number of physiological features (24 ET + 4 HR + 80 EEG = 108).
        """
        # Load the JSON data
        self.data = pd.read_json(json_path, orient="index")

        self.img_dir = img_dir
        self.use_sensorial = use_sensorial
        self.use_annotator_metadata = use_annotator_metadata
        self.transform = transform
        self.max_subjects = max_subjects
        self.n_samples = n_samples
        self.physio_dim = physio_dim

    def __len__(self):
        if self.n_samples is not None:
            return min(self.n_samples, len(self.data))
        return len(self.data)

    def _compute_marginals(self, item):
        """
        Computes the targets (Marginal Probabilities) based on annotator disagreement.
        """
        labels_2_1 = item["labels_task2_1"]
        labels_2_2 = item["labels_task2_2"]
        labels_2_3 = item["labels_task2_3"]

        # Task 2.1: Sexist Identification
        valid_2_1 = [l for l in labels_2_1 if l in ["YES", "NO"]]
        if len(valid_2_1) > 0:
            yes_count = valid_2_1.count("YES")
            t_2_1 = yes_count / len(valid_2_1)
        else:
            t_2_1 = 0.0

        # Conditional Mask
        # True if at least one annotator said YES (meaning 2.2 and 2.3 have valid signals)
        cond_mask = 1.0 if t_2_1 > 0.0 else 0.0

        # Task 2.2: Source Intention (DIRECT / JUDGEMENTAL)
        valid_2_2 = [l for l in labels_2_2 if l in ["DIRECT", "JUDGEMENTAL"]]
        if len(valid_2_2) > 0:
            judg_count = valid_2_2.count("JUDGEMENTAL")
            t_2_2 = judg_count / len(valid_2_2)
        else:
            t_2_2 = 0.0

        # Task 2.3: Sexism Categorization (Multi-Label)
        t_2_3 = np.zeros(len(TASK_2_3_CLASSES), dtype=np.float32)
        valid_2_3_annotators = 0

        for annotator_labels in labels_2_3:
            # Filter out the "-" and "UNKNOWN" labels
            valid_labels = [l for l in annotator_labels if l in TASK_2_3_CLASSES]
            if len(valid_labels) > 0:
                valid_2_3_annotators += 1
                for label in valid_labels:
                    idx = TASK_2_3_CLASSES.index(label)
                    t_2_3[idx] += 1.0

        if valid_2_3_annotators > 0:
            t_2_3 = t_2_3 / valid_2_3_annotators

        return (
            torch.tensor([t_2_1], dtype=torch.float32),
            torch.tensor([t_2_2], dtype=torch.float32),
            torch.tensor(t_2_3, dtype=torch.float32),
            torch.tensor([cond_mask], dtype=torch.float32),
        )

    def _extract_physio(self, item):
        """
        Extracts physiological data, handling missing modalities per subject
        and padding up to `max_subjects` (4).
        """
        physio_features = np.zeros(
            (self.max_subjects, self.physio_dim), dtype=np.float32
        )
        physio_mask = np.zeros(
            self.max_subjects, dtype=bool
        )  # True if subject data exists

        sensorial = item["sensorial"]
        users = sensorial["users"]
        modalities = sensorial["modalities"]

        for i, user in enumerate(users[: self.max_subjects]):
            physio_mask[i] = True
            user_features = []

            # We iterate through ET, HR, EEG and flatten their dictionaries.
            for mod in ["ET", "HR", "EEG"]:
                mod_data = modalities.get(mod, {}).get("by_user", {}).get(user, {})

                # Sort keys to ensure deterministic feature order
                sorted_keys = sorted(mod_data.keys())
                features = [mod_data[k] for k in sorted_keys]
                user_features.extend(features)

            # If the vector length doesn't match 108 exactly due to missing
            # modalities, we pad or truncate to self.physio_dim to prevent tensor shape errors.
            feat_len = len(user_features)
            if feat_len > 0:
                copy_len = min(feat_len, self.physio_dim)
                physio_features[i, :copy_len] = user_features[:copy_len]

        return torch.tensor(physio_features, dtype=torch.float32), torch.tensor(
            physio_mask, dtype=torch.bool
        )

    def _extract_annotators(self, item):
        """
        Extracts demographic data of the annotators who labeled this meme.
        """
        return {
            "gender": item["gender_annotators"],
            "age": item["age_annotators"],
            "country": item["countries_annotators"],
            "study_level": item["study_levels_annotators"],
            "ethnicity": item["ethnicities_annotators"],
        }

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # Image
        img_filename = item["path_memes"].split("/")[-1].split("\\")[-1]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Text
        text = item["text"]

        # Targets (Marginal probabilities)
        t_2_1, t_2_2, t_2_3, cond_mask = self._compute_marginals(item)

        # Build output dictionary
        sample = {
            "id": item["id_EXIST"],
            "split": item["split"],
            "image": image,
            "text": text,
            "target_2_1": t_2_1,
            "target_2_2": t_2_2,
            "target_2_3": t_2_3,
            "mask_conditional": cond_mask,
        }

        # Sensorial (Physiological) Data
        if self.use_sensorial:
            physio_features, physio_mask = self._extract_physio(item)
            sample["physio_features"] = physio_features
            sample["physio_mask"] = physio_mask

        # Annotator Metadata
        if self.use_annotator_metadata:
            sample["annotator_metadata"] = self._extract_annotators(item)

        return sample


if __name__ == "__main__":
    from torchvision import transforms

    # Define standard image transformation for testing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Instantiate dataset with ALL modalities toggled ON
    dataset = EXISTDataset(
        json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
        img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
        use_sensorial=True,
        use_annotator_metadata=True,
        transform=transform,
    )

    print(f"Dataset initialized with {len(dataset)} items.")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample ID: {sample['id']}")
        print(f"Text: {sample['text'][:50]}...")
        print(f"Image Tensor Shape: {sample['image'].shape}")

        print("\nTargets:")
        print(f"  Task 2.1 (YES prob): {sample['target_2_1'].item():.4f}")
        print(f"  Task 2.2 (JUDG prob): {sample['target_2_2'].item():.4f}")
        print(f"  Task 2.3 (Categories): {sample['target_2_3'].numpy()}")
        print(f"  Conditional Mask: {sample['mask_conditional'].item()}")

        if "physio_features" in sample:
            print(f"\nPhysiological Features Shape: {sample['physio_features'].shape}")
            print(f"Physiological Mask: {sample['physio_mask'].numpy()}")

        if "annotator_metadata" in sample:
            print(f"\nAnnotator Metadata: {sample['annotator_metadata']}")
