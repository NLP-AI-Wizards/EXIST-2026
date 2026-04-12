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
        max_subjects: int = 4,
        physio_dim: int = 108,
    ):
        # Load the JSON data
        self.data = pd.read_json(json_path, orient="index")

        # Reset index just in case the JSON keys aren't perfectly sequential
        self.data = self.data.reset_index(drop=True)

        self.img_dir = img_dir
        self.use_sensorial = use_sensorial
        self.use_annotator_metadata = use_annotator_metadata
        self.max_subjects = max_subjects
        self.physio_dim = physio_dim

    def __len__(self):
        return len(self.data)

    def _compute_marginals(self, item):
        labels_2_1 = item.get("labels_task2_1", [])
        labels_2_2 = item.get("labels_task2_2", [])
        labels_2_3 = item.get("labels_task2_3", [])

        # --- Task 2.1: Sexist Identification ---
        # Filter out "UNKNOWN" before counting
        valid_2_1 = [l for l in labels_2_1 if l in ["YES", "NO"]]
        t_2_1 = valid_2_1.count("YES") / len(valid_2_1) if len(valid_2_1) > 0 else 0.0

        # --- Task 2.2: Source Intention ---
        # Filter out "-" and "UNKNOWN"
        valid_2_2 = [l for l in labels_2_2 if l in ["DIRECT", "JUDGEMENTAL"]]
        t_2_2 = (
            valid_2_2.count("JUDGEMENTAL") / len(valid_2_2)
            if len(valid_2_2) > 0
            else 0.0
        )

        # --- Task 2.3: Sexism Categorization (Multi-Label) ---
        t_2_3 = np.zeros(len(TASK_2_3_CLASSES), dtype=np.float32)
        valid_2_3_annotators = 0

        for annotator_labels in labels_2_3:
            # Filter out "-" and "UNKNOWN"
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
        )

    def _extract_physio(self, item):
        physio_features = np.zeros(
            (self.max_subjects, self.physio_dim), dtype=np.float32
        )
        physio_mask = np.zeros(self.max_subjects, dtype=bool)

        # Safely extract sensorial data (handles Pandas NaN behavior)
        sensorial = item.get("sensorial", {})
        if pd.isna(sensorial):
            sensorial = {}

        users = sensorial.get("users", [])
        modalities = sensorial.get("modalities", {})

        for i, user in enumerate(users[: self.max_subjects]):
            physio_mask[i] = True
            user_features = []

            for mod in ["ET", "HR", "EEG"]:
                mod_data = modalities.get(mod, {}).get("by_user", {}).get(user, {})

                sorted_keys = sorted(mod_data.keys())
                features = [mod_data[k] for k in sorted_keys]
                user_features.extend(features)

            feat_len = len(user_features)
            if feat_len > 0:
                copy_len = min(feat_len, self.physio_dim)
                physio_features[i, :copy_len] = user_features[:copy_len]

        return torch.tensor(physio_features, dtype=torch.float32), torch.tensor(
            physio_mask, dtype=torch.bool
        )

    def _extract_annotators(self, item):
        return {
            "gender": item.get("gender_annotators", []),
            "age": item.get("age_annotators", []),
            "country": item.get("countries_annotators", []),
            "study_level": item.get("study_levels_annotators", []),
            "ethnicity": item.get("ethnicities_annotators", []),
        }

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # Image
        img_filename = item["path_memes"].split("/")[-1].split("\\")[-1]
        img_path = os.path.join(self.img_dir, img_filename)

        # Convert image to RGB array (handle missing image gracefully if needed)
        try:
            img_array = np.array(Image.open(img_path).convert("RGB"))
        except FileNotFoundError:
            # Fallback to black image if file is missing
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)

        text = item.get("text", "")

        t_2_1, t_2_2, t_2_3 = self._compute_marginals(item)

        # Build output dictionary
        sample = {
            "id": item["id_EXIST"],
            "image": img_array,
            "text": text,
            "target_2_1": t_2_1,
            "target_2_2": t_2_2,
            "target_2_3": t_2_3,
            "mask_conditional": torch.tensor(
                [1.0 if t_2_1.item() > 0.0 else 0.0], dtype=torch.float32
            ),
        }

        # Sensorial Data
        if self.use_sensorial:
            physio_features, physio_mask = self._extract_physio(item)
            sample["physio_features"] = physio_features
            sample["physio_mask"] = physio_mask

        # Annotator Metadata
        if self.use_annotator_metadata:
            sample["annotator_metadata"] = self._extract_annotators(item)

        return sample


if __name__ == "__main__":
    dataset = EXISTDataset(
        json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
        img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
        use_sensorial=True,
        use_annotator_metadata=True,
    )

    print(f"Dataset initialized with {len(dataset)} items.")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample ID: {sample['id']}")
        print(f"Text: {sample['text'][:50]}...")
        print(f"Image Array Shape: {sample['image'].shape}")

        print("\nTargets:")
        print(f"  Task 2.1 (YES prob): {sample['target_2_1'].item():.4f}")
        print(f"  Task 2.2 (JUDG prob): {sample['target_2_2'].item():.4f}")
        print(f"  Task 2.3 (Categories): {sample['target_2_3'].numpy()}")
        print(f"  Conditional Mask: {sample['mask_conditional'].item()}")

        if "physio_features" in sample:
            print(f"\nPhysiological Features Shape: {sample['physio_features'].shape}")
            print(f"Physiological Mask: {sample['physio_mask'].numpy()}")
