# Step 1: Import required libraries

import pandas as pd          # for data handling
import numpy as np           # for numerical operations

# Libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------

# Step 2: Load the dataset

data = pd.read_csv("insurance_data_linear.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# ------------------------------------------------------------

# Step 3: Understand dataset structure

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# ------------------------------------------------------------

# Step 4: Check for missing values

print("\nMissing values in dataset:")
print(data.isnull().sum())

# If missing values existed we could fill them like:
# data['age'].fillna(data['age'].mean(), inplace=True)

# ------------------------------------------------------------

# Step 5: Encode categorical variables

# Columns like sex, smoker, region contain text
# Machine learning models require numbers

data = pd.get_dummies(data, drop_first=True)

print("\nDataset after encoding categorical variables:")
print(data.head())

# ------------------------------------------------------------

# Step 6: Separate features (X) and target (y)

# 'charges' is the value we want to predict

X = data.drop("charges", axis=1)   # independent variables
y = data["charges"]                # dependent variable

# ------------------------------------------------------------

# Step 7: Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% data for testing
    random_state=42     # ensures reproducible results
)

# ------------------------------------------------------------

# Step 8: Feature Scaling

# Scaling makes all values comparable and improves model performance

scaler = StandardScaler()

# Fit scaler only on training data
X_train = scaler.fit_transform(X_train)

# Apply same scaling to test data
X_test = scaler.transform(X_test)

# ------------------------------------------------------------

# Step 9: Create Linear Regression model

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# ------------------------------------------------------------

# Step 10: Make predictions using test data

y_pred = model.predict(X_test)

# ------------------------------------------------------------

# Step 11: Evaluate model performance

print("\nModel Evaluation")

# R2 Score (how well the model fits the data)
print("R2 Score:", r2_score(y_test, y_pred))

# Mean Squared Error (prediction error)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# ------------------------------------------------------------

# Step 12: Compare actual vs predicted values

results = pd.DataFrame({
    "Actual Charges": y_test,
    "Predicted Charges": y_pred
})

print("\nSample Predictions:")
print(results.head())