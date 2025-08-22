import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load dataset
df = pd.read_csv("/kaggle/input/titanic/train.csv")

# Features & target
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"
X = df[features].copy()
y = df[target].copy()

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define preprocessing
num_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
cat_features = ["Sex", "Embarked"]

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_features),
    ("cat", categorical_pipe, cat_features),
])

# Model
rf = RandomForestClassifier(
    n_estimators=400, min_samples_split=4, min_samples_leaf=2,
    n_jobs=-1, random_state=42
)

clf = Pipeline([
    ("prep", preprocess),
    ("model", rf),
])

# Train and save
clf.fit(X_train, y_train)
dump(clf, "titanic_model.joblib")
print("✅ Model trained and saved as titanic_model.joblib")
