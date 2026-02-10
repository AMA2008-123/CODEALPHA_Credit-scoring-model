import numpy as np
import joblib

# Load model & scaler
model = joblib.load("Random_Forest.pkl")
scaler = joblib.load("scaler.pkl")

# Example new customer data
# (Use same feature order as dataset)

sample = np.array([[
    0.45,   # RevolvingUtilizationOfUnsecuredLines
    35,     # age
    2,      # NumberOfTime30-59DaysPastDueNotWorse
    0.30,   # DebtRatio
    50000,  # MonthlyIncome
    5,      # NumberOfOpenCreditLinesAndLoans
    0,      # NumberOfTimes90DaysLate
    1,      # NumberRealEstateLoansOrLines
    0,      # NumberOfTime60-89DaysPastDueNotWorse
    2,      # NumberOfDependents
]])

# Feature engineering (same as training)
debt_income_ratio = sample[:, 3] / sample[:, 4]
sample = np.hstack((sample, debt_income_ratio.reshape(-1, 1)))

# Scale
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

# Result
if prediction == 1:
    print("⚠️ Customer is HIGH credit risk (Likely to default)")
else:
    print("✅ Customer is CREDITWORTHY")

print(f"Default Probability: {probability*100:.2f}%")
