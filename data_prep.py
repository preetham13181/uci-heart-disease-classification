import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. Load and combine the four processed UCI cohorts
# ------------------------------------------------------------

urls = {
    "cleveland": (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/heart-disease/processed.cleveland.data"
    ),
    "hungarian": (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/heart-disease/processed.hungarian.data"
    ),
    "switzerland": (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/heart-disease/processed.switzerland.data"
    ),
    "va": (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/heart-disease/processed.va.data"
    ),
}

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

cohorts = []

for source, url in urls.items():
    cohort = pd.read_csv(
        url,
        header=None,
        names=columns,
        na_values="?",
        skipinitialspace=True
    )
    cohort["source"] = source
    cohorts.append(cohort)

data = pd.concat(cohorts, ignore_index=True)

# Convert the 14 UCI fields to numeric; leave source as text.
for column in columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# The outcome must be present. Preserve missing predictor values for now.
data = data.dropna(subset=["num"]).copy()
data["num"] = data["num"].round().astype(int)

# ------------------------------------------------------------
# 2. Check the actual combined-data distribution
# ------------------------------------------------------------

print("Combined data shape:", data.shape)
print("\nCounts for the original num outcome:")
print(data["num"].value_counts().sort_index())

print("\nProportions for the original num outcome:")
print(
    data["num"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)

# Binary task: num == 0 means no disease; num > 0 means disease present.
data["target"] = (data["num"] > 0).astype("int8")

print("\nBinary target proportions (0=no disease, 1=disease):")
print(data["target"].value_counts(normalize=True).sort_index().round(4))

# ------------------------------------------------------------
# 3. Split real records before generating anything
# ------------------------------------------------------------

train_real, test_real = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["num"]
)

train_real = train_real.copy()
test_real = test_real.copy()

print("\nReal training rows:", len(train_real))
print("Real test rows:", len(test_real))

# ------------------------------------------------------------
# 4. Generate synthetic rows from real training rows only
# ------------------------------------------------------------

target_training_size = 10_000
n_synthetic = max(0, target_training_size - len(train_real))

# Only these measured quantities receive small numeric perturbations.
# Other fields, including categorical codes and source, are copied from
# the sampled donor row.
continuous_columns = [
    column for column in ["age", "trestbps", "chol", "thalach", "oldpeak"]
    if column in train_real.columns
]

integer_like_columns = [
    column for column in ["age", "trestbps", "chol", "thalach"]
    if column in train_real.columns
]

rng = np.random.default_rng(42)
class_counts = train_real["num"].value_counts().sort_index()
class_probabilities = class_counts / class_counts.sum()

# Allocate generated rows to match the real training-set proportions
# of num = 0, 1, 2, 3, and 4.
raw_quotas = class_probabilities * n_synthetic
quotas = np.floor(raw_quotas).astype(int)
rows_left = int(n_synthetic - quotas.sum())

remainders = (raw_quotas - quotas).sort_values(ascending=False)
for label in remainders.index[:rows_left]:
    quotas.loc[label] += 1

noise_fraction = 0.02
synthetic_parts = []

for label, n_rows in quotas.items():
    if n_rows == 0:
        continue

    donor_pool = train_real.loc[train_real["num"] == label]

    donor_indices = rng.choice(
        donor_pool.index.to_numpy(),
        size=int(n_rows),
        replace=True
    )
    generated = donor_pool.loc[donor_indices].copy().reset_index(drop=True)

    # Add small noise to the selected continuous fields only.
    # Clip values to the range observed for that class in the training data.
    for column in continuous_columns:
        pool_values = pd.to_numeric(donor_pool[column], errors="coerce")
        generated_values = pd.to_numeric(generated[column], errors="coerce")
        class_sd = pool_values.std()

        if pd.notna(class_sd) and class_sd > 0:
            observed = generated_values.notna()
            generated_values.loc[observed] += rng.normal(
                loc=0,
                scale=noise_fraction * class_sd,
                size=int(observed.sum())
            )

            generated[column] = generated_values.clip(
                lower=pool_values.min(),
                upper=pool_values.max()
            )

    for column in integer_like_columns:
        generated[column] = pd.to_numeric(
            generated[column],
            errors="coerce"
        ).round()

    # Preserve the donor's num value and derive the binary target from it.
    generated["target"] = (generated["num"] > 0).astype("int8")
    generated["is_synthetic"] = 1
    synthetic_parts.append(generated)

if synthetic_parts:
    synthetic = pd.concat(synthetic_parts, ignore_index=True)
else:
    synthetic = pd.DataFrame(
        columns=list(train_real.columns) + ["is_synthetic"]
    )

# Mark original rows, then combine them with the synthetic rows.
train_real["is_synthetic"] = 0
test_real["is_synthetic"] = 0

train_augmented = pd.concat(
    [train_real, synthetic],
    ignore_index=True
)

# ------------------------------------------------------------
# 5. Review the results and save the files
# ------------------------------------------------------------

print("\nOriginal real training rows:", len(train_real))
print("Synthetic training rows:", len(synthetic))
print("Augmented training rows:", len(train_augmented))
print("Real test rows:", len(test_real))

print("\nAugmented training num proportions:")
print(
    train_augmented["num"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)

print("\nAugmented binary target proportions:")
print(
    train_augmented["target"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)

output_directory = Path("outputs")
output_directory.mkdir(exist_ok=True)

synthetic.to_csv(
    output_directory / "uci_synthetic_training_rows.csv",
    index=False
)
train_augmented.to_csv(
    output_directory / "uci_augmented_training_10000.csv",
    index=False
)
test_real.to_csv(
    output_directory / "uci_real_test_holdout.csv",
    index=False
)

print("\nFiles saved in the outputs folder.")

# ------------------------------------------------------------
# 6. Prepare predictors and labels for later model training
# ------------------------------------------------------------

# Do not use the original outcome, source metadata, or synthetic flag
# as model predictors.
columns_to_exclude = ["num", "target", "source", "is_synthetic"]

X_train = train_augmented.drop(
    columns=columns_to_exclude,
    errors="ignore"
)
y_train = train_augmented["target"]

X_test = test_real.drop(
    columns=columns_to_exclude,
    errors="ignore"
)
y_test = test_real["target"]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
