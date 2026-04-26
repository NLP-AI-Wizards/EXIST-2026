import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA

from dataset import EXISTDataset

TASK_2_2_CLASSES = ["DIRECT", "JUDGEMENTAL"]

TASK_2_3_CLASSES = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]

dataset = EXISTDataset(
    json_path=r"data/EXIST 2026 Memes Dataset/training/EXIST2026_training.json",
    img_dir=r"data/EXIST 2026 Memes Dataset/training/memes",
    include_image=False,
    include_text=False,
    include_id=False,
    use_sensorial=True,
    max_subjects=4,
)
eeg_rows = []
y_2_1_rows = []
y_2_2_rows = []
y_2_3_rows = []

for i in range(len(dataset)):
    sample = dataset[i]
    eeg_features = sample["eeg_features"].numpy()  # (max_subjects, 80)
    physio_mask = sample["physio_mask"].numpy()  # (max_subjects,)

    # Keep one row per available subject
    valid_eeg = eeg_features[physio_mask]
    if len(valid_eeg) == 0:
        continue

    target_2_1 = sample["target_2_1"].item()
    target_2_2 = sample["target_2_2"].item()
    target_2_3 = sample["target_2_3"].numpy()

    eeg_rows.append(valid_eeg)
    y_2_1_rows.append(np.full(len(valid_eeg), target_2_1, dtype=np.float32))
    y_2_2_rows.append(np.full(len(valid_eeg), target_2_2, dtype=np.float32))
    y_2_3_rows.append(np.repeat(target_2_3[None, :], len(valid_eeg), axis=0))

X_eeg = np.concatenate(eeg_rows, axis=0)
y_2_1 = np.concatenate(y_2_1_rows, axis=0)
y_2_2 = np.concatenate(y_2_2_rows, axis=0)
y_2_3 = np.concatenate(y_2_3_rows, axis=0)

print(X_eeg.shape)
print(y_2_1.shape)
print(y_2_2.shape)
print(y_2_3.shape)


# We only care about memes that are generally considered sexist - P(YES) >= 0.5)
sexist_mask = y_2_1 >= 0.5
X_eeg_sexist = X_eeg[sexist_mask]
y_2_2_sexist = y_2_2[sexist_mask]
y_2_3_sexist = y_2_3[sexist_mask]

dominant_class_2_2_idx = (y_2_2_sexist >= 0.5).astype(
    int
)  # 0 for DIRECT, 1 for JUDGEMENTAL
dominant_class_2_2_names = [TASK_2_2_CLASSES[i] for i in dominant_class_2_2_idx]

# MANOVA requires mutually exclusive groups. We assign each meme to its dominant category.
dominant_class_2_3_idx = np.argmax(y_2_3_sexist, axis=1)
dominant_class_2_3_names = [TASK_2_3_CLASSES[i] for i in dominant_class_2_3_idx]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MANOVA on EEG features")
    parser.add_argument(
        "--task",
        choices=["2.2", "2.3"],
        required=True,
        help="Which task's categories to analyze",
    )
    parser.add_argument(
        "--ncomponents",
        type=float,
        default=0.95,
        help="Number of PCA components to keep",
    )
    args = parser.parse_args()
    print(f"Analyzing {len(X_eeg_sexist)} sexist memes...")

    # We use PCA to extract orthogonal components that explain 95% of the brain variance.
    pca = PCA(n_components=args.ncomponents, random_state=42)
    X_eeg_pca = pca.fit_transform(X_eeg_sexist)

    n_components = X_eeg_pca.shape[1]
    print(
        f"PCA reduced 80 collinear EEG features to {n_components} independent components "
        f"explaining {pca.explained_variance_ratio_.sum():.2%} of the variance."
    )

    df = pd.DataFrame(X_eeg_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
    if args.task == "2.2":
        df["Category"] = dominant_class_2_2_names
    else:
        df["Category"] = dominant_class_2_3_names

    dependent_vars = " + ".join([f"PC{i + 1}" for i in range(n_components)])
    formula = f"{dependent_vars} ~ C(Category)"

    print("\nRunning MANOVA...")
    manova = MANOVA.from_formula(formula, data=df)
    result = manova.mv_test()

    p_value = result.results["C(Category)"]["stat"].loc["Wilks' lambda", "Pr > F"]

    print("=" * 60)
    print("MANOVA RESULTS")
    print("=" * 60)
    print(result.summary())
    print("=" * 60)

    if p_value < 0.05:
        print(f"p-value = {p_value:.4f}. The result IS statistically significant.")
    else:
        print(f"p-value = {p_value:.4f}. The result IS NOT statistically significant.")
