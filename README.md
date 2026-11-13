# UCI Heart Disease Classification

An exploratory machine-learning project investigating whether structured clinical attributes can classify the **presence of heart disease** in the UCI Heart Disease dataset. The project also compares training with real data alone against training with a synthetically augmented set. It is for educational research and is **not intended for clinical use**.

## Goal

Train and evaluate binary classifiers for heart-disease presence, prioritizing sensitivity (recall) while also reporting other relevant metrics. The project will examine whether simple synthetic-data augmentation changes model performance on held-out real records.

## Dataset

The project combines the four processed regional datasets distributed with the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease): Cleveland, Hungarian, Switzerland, and VA.

- **Real records:** 920
- **Original fields:** 13 input attributes and the `num` outcome (14 columns total)
- **Binary target:** `num = 0` is no disease; `num > 0` is disease present
- **Missing data:** Several input attributes contain missing values and require appropriate preprocessing

The regional records are pooled for the current experiment. A random train/test split across the pooled records evaluates performance on held-out records from these cohorts; it does not establish performance at a new hospital or in a new population.

## Methods

This is a supervised binary-classification task. Planned models are:

- Random Forest
- XGBoost
- Logistic regression as a baseline

Recall/sensitivity is a priority, but model performance will also be assessed with measures such as specificity, precision, F1 score, ROC-AUC, and precision-recall AUC. Any decision threshold will be selected using training/validation data, not the final test set.

## Work completed

- Loaded and combined the four processed UCI regional datasets.
- Inspected the `num` outcome distribution and defined the binary target as `num > 0`.
- Split the real data into **736 training records** and **184 held-out test records**.
- Generated **9,264 synthetic training rows** using a simple class-conditioned smoothed-bootstrap procedure: donor rows are sampled from the real training data within the same `num` class, and small noise is added to selected continuous measurements. Categorical/discrete values are copied from donor rows.
- Combined the real training rows and synthetic rows into a **10,000-row augmented training set**. The held-out test set remains real only.

The generated rows are derived from the training records; they are **not additional independent patients** and do not increase the amount of real clinical evidence. The augmentation method is exploratory and is not a validated clinical data-generation method.

## Next steps

1. Build preprocessing pipelines that handle missing values and encode nominal categorical features appropriately. Fit preprocessing using training data only.
2. Train and tune the planned models without using the held-out real test set for model selection.
3. Compare real-only training against real-plus-synthetic training on the same held-out real records.
4. Report recall alongside specificity, precision, F1, ROC-AUC, and precision-recall AUC; interpret results cautiously because the real dataset is small.
5. Consider a site-held-out evaluation to explore how performance changes across source cohorts.
6. Document limitations and keep the project framed as exploratory—not as a clinical decision tool.

## Team

- Hemanth
- Jonhatan
- Preetham

