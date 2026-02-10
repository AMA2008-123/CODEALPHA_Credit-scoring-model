import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("cs-training.csv")

# Drop index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Handle missing values (same as training)
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

# Replace zero income
df["MonthlyIncome"] = df["MonthlyIncome"].replace(0, df["MonthlyIncome"].median())

# Feature engineering
df["DebtIncomeRatio"] = df["DebtRatio"] / df["MonthlyIncome"]

# Remove infinity
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN again
df.fillna(df.median(), inplace=True)

# Features & target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# Load scaler & model
scaler = joblib.load("scaler.pkl")
model = joblib.load("Random_Forest.pkl")

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))
print("ROC-AUC:", roc_auc_score(y, y_prob))

print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
