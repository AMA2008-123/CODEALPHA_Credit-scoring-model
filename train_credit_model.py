import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

## Load dataset
df = pd.read_csv("cs-training.csv")

# Drop index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Handle missing values properly
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

# Replace 0 income to avoid division error
df["MonthlyIncome"] = df["MonthlyIncome"].replace(0, df["MonthlyIncome"].median())

# Feature Engineering
df["DebtIncomeRatio"] = df["DebtRatio"] / df["MonthlyIncome"]

# Replace infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill any new NaN
df.fillna(df.median(), inplace=True)

# Features & Target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train all models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")
    print(f"✅ {name} trained & saved")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("🎯 Training completed.")
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
