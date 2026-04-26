import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
        include_image: bool = True,
        include_text: bool = True,
        include_id: bool = True,
        use_sensorial: bool = False,
        max_subjects: int = 4,
    ):
        # Load the JSON data
        self.data = pd.read_json(json_path, orient="index")

        # Reset index just in case the JSON keys aren't perfectly sequential
        self.data = self.data.reset_index(drop=True)

        self.img_dir = img_dir
        self.include_image = include_image
        self.include_text = include_text
        self.include_id = include_id
        self.use_sensorial = use_sensorial
        self.max_subjects = max_subjects
        self.physio_dim = 108

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

    def _extract_image(self, item):
        img_filename = item["path_memes"].split("/")[-1].split("\\")[-1]
        img_path = os.path.join(self.img_dir, img_filename)
        # Convert first to RGBA then to RGB
        img = Image.open(img_path)
        img_array = np.array(img.convert("RGBA").convert("RGB"))
        return img_array

    def _extract_text(self, item):
        return item.get("text", "")

    def _extract_physio(self, item):
        # The exact expected feature counts based on the EXIST guidelines
        dim_et = 24
        dim_hr = 4
        dim_eeg = 80

        et_features = np.zeros((self.max_subjects, dim_et), dtype=np.float32)
        hr_features = np.zeros((self.max_subjects, dim_hr), dtype=np.float32)
        eeg_features = np.zeros((self.max_subjects, dim_eeg), dtype=np.float32)
        physio_mask = np.zeros(self.max_subjects, dtype=bool)

        sensorial = item.get("sensorial", {})
        if pd.isna(sensorial):
            sensorial = {}

        users = sensorial.get("users",[])
        modalities = sensorial.get("modalities", {})

        for i, user in enumerate(users[: self.max_subjects]):
            physio_mask[i] = True

            # 1. Extract ET (Eye Tracking)
            et_data = modalities.get("ET", {}).get("by_user", {}).get(user, {})
            et_vals =[et_data[k] for k in sorted(et_data.keys())]
            if et_vals:
                et_features[i, :min(len(et_vals), dim_et)] = et_vals[:dim_et]

            # 2. Extract HR (Heart Rate)
            hr_data = modalities.get("HR", {}).get("by_user", {}).get(user, {})
            hr_vals = [hr_data[k] for k in sorted(hr_data.keys())]
            if hr_vals:
                hr_features[i, :min(len(hr_vals), dim_hr)] = hr_vals[:dim_hr]

            # 3. Extract EEG
            eeg_data = modalities.get("EEG", {}).get("by_user", {}).get(user, {})
            eeg_vals =[eeg_data[k] for k in sorted(eeg_data.keys())]
            if eeg_vals:
                eeg_features[i, :min(len(eeg_vals), dim_eeg)] = eeg_vals[:dim_eeg]

        return (
            torch.tensor(et_features, dtype=torch.float32),
            torch.tensor(hr_features, dtype=torch.float32),
            torch.tensor(eeg_features, dtype=torch.float32),
            torch.tensor(physio_mask, dtype=torch.bool),
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

        image = self._extract_image(item) if self.include_image else None
        text = self._extract_text(item) if self.include_text else None

        t_2_1, t_2_2, t_2_3 = self._compute_marginals(item)

        # Build output dictionary
        sample = {
            "target_2_1": t_2_1,
            "target_2_2": t_2_2,
            "target_2_3": t_2_3,
            "mask_conditional": torch.tensor(
                [1.0 if t_2_1.item() > 0.0 else 0.0], dtype=torch.float32
            ),
        }

        if self.include_id:
            sample["id"] = item["id_EXIST"]
        if self.include_image:
            sample["image"] = image
        if self.include_text:
            sample["text"] = text

        # Sensorial Data
        if self.use_sensorial:
            et_feat, hr_feat, eeg_feat, physio_mask = self._extract_physio(item)
            sample["et_features"] = et_feat
            sample["hr_features"] = hr_feat
            sample["eeg_features"] = eeg_feat
            sample["physio_mask"] = physio_mask

        return sample
