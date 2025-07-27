# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load dataset (Kaggle auto mounts /kaggle/input)
df = pd.read_csv("/kaggle/input/titanic/train.csv")

# Step 3: Preprocessing - drop columns we won't use and handle missing values
df = df.copy()
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # Drop unused / non-numeric columns

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 4: Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
