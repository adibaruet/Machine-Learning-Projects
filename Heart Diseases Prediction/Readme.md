# Heart Attack Possibility Prediction

A machine learning project that predicts whether a patient has a **higher or lower chance of heart disease** from 13 common clinical measurements. Four classifiers (KNN, MLP, AdaBoost, Random Forest) are trained, compared, and combined into a voting ensemble.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Purpose-Educational-green)

> **Disclaimer:** This is a learning project. It is not a medical device and must not be used to diagnose or make health decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Models](#models)
5. [Results](#results)
6. [Key Findings](#key-findings)
7. [Getting Started](#getting-started)
8. [Using the Saved Model](#using-the-saved-model)
9. [Repository Structure](#repository-structure)
10. [Limitations](#limitations)
11. [Future Work](#future-work)
12. [Author](#author)

---

## Overview

Heart disease is one of the leading causes of death worldwide, and early risk screening can help. This project asks a simple question:

> Given a patient's age, blood pressure, cholesterol, chest pain type, ECG results and similar values, can a model tell if the chance of heart disease is higher or lower?

The notebook walks through the full pipeline: loading data, exploring it, preparing it without data leakage, training four models, comparing them with several metrics, and saving the best model for reuse.

---

## Dataset

- **Source:** [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (Cleveland subset, as shared on Kaggle)
- **Size:** 303 patients, 13 input features, 1 target column
- **Missing values:** none
- **Class balance:** 165 patients with higher chance (1), 138 with lower chance (0)

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `restecg` | Resting ECG result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia test result |
| `target` | **0 = less chance, 1 = more chance** of heart disease |

---

## Project Workflow

1. **Load data:** the notebook finds `heart.csv` automatically under `/kaggle/input`.
2. **Explore:** `info()`, `describe()`, and a class distribution plot.
3. **Split first, then scale:** an 80/20 stratified train/test split, then `StandardScaler` fitted on the training set only. This keeps the test set unseen and avoids data leakage.
4. **Train and validate:** each model is checked with 5-fold cross-validation on the training set, then evaluated once on the held-out test set.
5. **Evaluate:** accuracy, precision, recall, F1, confusion matrix, and ROC-AUC (computed from predicted probabilities).
6. **Compare:** summary table, bar charts, and one combined ROC plot for all models.
7. **Interpret:** Random Forest feature importance.
8. **Ensemble:** a soft-voting classifier that averages the four models' probabilities.
9. **Predict and save:** run a prediction for one example patient and save the model with `joblib`.

A fixed random seed (`SEED = 42`) is used everywhere, so the results are reproducible.

---

## Models

| Model | Main settings |
|---|---|
| **K-Nearest Neighbors** | `n_neighbors=7`, `weights='distance'`, `metric='manhattan'` |
| **MLP (neural network)** | ReLU, Adam solver, `alpha=0.0001`, `max_iter=2600` |
| **AdaBoost** | `n_estimators=250`, `learning_rate=0.01` |
| **Random Forest** | `n_estimators=100`, `max_features='sqrt'`, `criterion='gini'` |
| **Voting Ensemble** | Soft voting over all four models above |

---

## Results

Evaluated on the held-out test set of 61 patients.

| Model | CV Accuracy (train) | Test Accuracy | AUC |
|---|---|---|---|
| KNN | 0.818 | 0.754 | 0.896 |
| MLP | 0.801 | 0.721 | 0.835 |
| AdaBoost | 0.830 | 0.820 | 0.892 |
| **Random Forest** | **0.830** | **0.836** | **0.909** |
| Voting Ensemble | – | 0.803 | 0.897 |

**Random Forest** gave the best test accuracy and AUC.

Per-class results for Random Forest:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Less chance | 0.95 | 0.68 | 0.79 |
| More chance | 0.78 | 0.97 | 0.86 |

### Top features (Random Forest importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `cp` (chest pain type) | 0.157 |
| 2 | `thalach` (max heart rate) | 0.117 |
| 3 | `oldpeak` (ST depression) | 0.113 |
| 4 | `thal` (thalassemia) | 0.107 |
| 5 | `chol` (cholesterol) | 0.087 |

---

## Key Findings

- Random Forest was the strongest single model, reaching about 84% test accuracy and 0.91 AUC.
- The ensemble did not beat Random Forest here, which can happen when some members (like the MLP) are weaker than the best model.
- The models are very good at catching patients with a higher chance (recall of 0.91–0.97) but miss more of the lower-chance patients (recall of 0.57–0.68). In other words, they lean toward predicting "more chance".
- Chest pain type, maximum heart rate, and ST depression were the most influential features. Fasting blood sugar and resting ECG mattered least.
- With only 61 test patients, one patient equals about 1.6 percentage points, so small differences between models are not strong evidence.

---

## Getting Started

### Option 1: Run on Kaggle (easiest)

1. Create a new Kaggle notebook and upload `heart-diseases-prediction.ipynb` (**File → Upload notebook**).
2. Click **Add Input** and add the heart disease dataset containing `heart.csv`.
3. Click **Run All**.

The notebook finds the CSV path by itself, so no path editing is needed.

### Option 2: Run locally
```bash
git clone https://github.com/adibaruet/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Heart\ Diseases\ Prediction
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook heart-diseases-prediction.ipynb
```


Before running locally, change these two lines in the notebook:

```python
# Data loading cell
df = pd.read_csv("heart.csv")          # instead of the /kaggle/input path

# Model saving cell
joblib.dump(..., "heart_model.joblib") # instead of /kaggle/working/...
```

Place `heart.csv` in the same folder as the notebook.

### Requirements

- Python 3.9 or newer
- numpy, pandas, matplotlib, seaborn
- scikit-learn (recent version; `max_features='sqrt'` is used because `'auto'` was removed)
- joblib

---

## Using the Saved Model

The notebook saves the model, the scaler, and the column order together in one file. To reuse it:

```python
import joblib
import pandas as pd

bundle = joblib.load("heart_model.joblib")
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]

patient = pd.DataFrame([{
    "age": 54, "sex": 1, "cp": 2, "trestbps": 130, "chol": 246,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}])[columns]

scaled = pd.DataFrame(scaler.transform(patient), columns=columns)
prob = model.predict_proba(scaled)[0, 1]

print("More chance" if prob >= 0.5 else "Less chance", f"({prob:.1%})")
```

The example values above are made up and only show the input format.

---

## Repository Structure

```
.
├── heart-diseases-prediction.ipynb   # Full notebook: EDA, training, evaluation, saving
├── heart.csv                         # Dataset (optional, if you want to run locally)
├── heart_model.joblib                # Saved model + scaler (created after running)
└── README.md
```

---

## Limitations

- **Small dataset:** only 303 patients (302 after removing one duplicate row), so results can change with a different split.
- **Duplicate row:** one duplicate row was found at the end of the notebook. The reported results were produced before it was removed, and the effect is negligible, but for the cleanest workflow the duplicate check should run right after loading the data.
- **Single split:** the final numbers come from one train/test split. Cross-validation on the full data would give a steadier estimate.
- **No hyperparameter tuning:** model settings are reasonable defaults, not tuned values.
- **Data origin:** the data comes from one hospital study and may not represent other populations.
- **Not clinical software:** the output is a probability from a student project, not medical advice.

---

## Future Work

- Move the duplicate check to right after loading the data
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Report cross-validated mean ± standard deviation for every model
- Try gradient boosting models such as LightGBM or XGBoost
- Adjust the decision threshold to improve recall on the "less chance" class
- Add SHAP values for per-patient explanations
- Build a small Streamlit web app around the saved model

---

## Author

**Humaira Tasnim Adiba**
B.Sc. in Electrical and Computer Engineering, Rajshahi University of Engineering and Technology (RUET)
GitHub: [@adibaruet](https://github.com/adibaruet)

---

## Acknowledgements

- UCI Machine Learning Repository for the Heart Disease dataset
- The scikit-learn community for the tools used throughout this project
