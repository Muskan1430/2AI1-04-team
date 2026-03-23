# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# 2. LOAD DATA
df = pd.read_csv("insurance_data_linear.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)



# 3. DATA CLEANING
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing values:")
print(df.isnull().sum())


# 4. HANDLE CATEGORICAL DATA
df = pd.get_dummies(df, drop_first=True)

print("\nColumns after encoding:")
print(df.columns)


# 5. DEFINE FEATURES AND TARGET
X = df.drop("charges", axis=1)
y = df["charges"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# 6. FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. POLYNOMIAL FEATURES
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

print("\nShape after polynomial features:", X_poly.shape)


# 8. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)


# 9. MODEL TRAINING (Simple Ridge)
model = Ridge()  # no tuning
model.fit(X_train, y_train)


# 10. CROSS-VALIDATION
cv_score = cross_val_score(model, X_poly, y, cv=5).mean()
print("\nCross-validation score:", cv_score)


# 11. FINAL EVALUATION
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# 12. SAMPLE PREDICTIONS
print("\nSample Predictions:")
print(y_pred[:5])
