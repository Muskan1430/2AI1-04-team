# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# 2. LOAD AND EXPLORE DATASET
# =====================================
df = pd.read_csv("insurance_data_linear.csv")

print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

# =====================================
# 3. DATA PREPROCESSING
# =====================================

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# =====================================
# 4. TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 5. FEATURE SCALING
# =====================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# 6. BASE MODEL (LINEAR REGRESSION)
# =====================================
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

print("\n--- Linear Regression ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# =====================================
# 7. SIMPLE MODEL IMPROVEMENTS
# =====================================

#  (A) Cross Validation
cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="r2")
print("\nCross-validation R2 scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))


#  (B) Regularization (Ridge Regression)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

print("\n--- Ridge Regression (Improved) ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R2 Score:", r2_score(y_test, y_pred_ridge))


#  (C) Feature Engineering (Simple)
# Example: create new feature if bmi exists
if "bmi" in df.columns:
    df["bmi_squared"] = df["bmi"] ** 2

    X = df.drop("charges", axis=1)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr.fit(X_train_scaled, y_train)
    y_pred_new = lr.predict(X_test_scaled)

    print("\n--- After Feature Engineering ---")
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_new)))
    print("R2 Score:", r2_score(y_test, y_pred_new))


# =====================================
# 8. FINAL SUMMARY
# =====================================
print("\nModel Improvement Techniques Applied:")
print("✔ Cross Validation")
print("✔ Regularization (Ridge)")
print("✔ Feature Engineering")
