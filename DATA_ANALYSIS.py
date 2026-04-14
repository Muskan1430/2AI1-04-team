import pandas as pd

# STEP 1: Load dataset
df = pd.read_csv("insurance_data_linear.csv")

print("First 5 rows:")
print(df.head())

# STEP 2: Convert text to numbers
df = pd.get_dummies(df, drop_first=True)

# STEP 3: Split input and output
X = df.drop("charges", axis=1)
y = df["charges"]

# STEP 4: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 5: Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Predict
y_pred = model.predict(X_test)

# STEP 7: Evaluate
from sklearn.metrics import r2_score
print("Accuracy (R2 Score):", r2_score(y_test, y_pred))