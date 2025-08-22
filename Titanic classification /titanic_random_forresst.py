# Step 0: Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# Step 1: Load
df = pd.read_csv("/kaggle/input/titanic/train.csv")

# Step 2: Select features (keep it simple & robust)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"
X = df[features].copy()
y = df[target].copy()

# Step 3: Train/validation split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Preprocess with ColumnTransformer (no manual get_dummies)
num_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]  # Pclass numeric is fine
cat_features = ["Sex", "Embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_features),
    ("cat", categorical_pipe, cat_features),
])

# Step 5: Model (solid baseline)
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

clf = Pipeline([
    ("prep", preprocess),
    ("model", rf),
])

# Step 6: Fit
clf.fit(X_train, y_train)

# Step 7: Evaluate
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {auc:.4f}\n")
print("Classification report:\n", classification_report(y_test, pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# Step 8 (optional): Quick cross-validation on the training set
cv_auc = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"\nCV ROC AUC (5-fold): mean={cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Step 9 (optional): Feature importance via permutation (on the test split)
# Build readable feature names after one-hot
ohe = clf.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_out = list(ohe.get_feature_names_out(cat_features))
final_feature_names = num_features + cat_out

perm = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = perm.importances_mean.argsort()[::-1]

print("\nPermutation feature importance (top 10):")
for i in sorted_idx[:10]:
    print(f"{final_feature_names[i]:25s}  {perm.importances_mean[i]:.5f}")
