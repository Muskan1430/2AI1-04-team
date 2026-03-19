import pandas as pd

# -------------------------------
# 1. Load the dataset
# -------------------------------
file_path = "insurance_data_linear.csv"
df = pd.read_csv(file_path)

# -------------------------------
# 2. Display basic information
# -------------------------------
print("Dataset Shape:", df.shape)
print("\nColumns in dataset:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# 3. Define target and features
# -------------------------------
target_column = "charges"

# Features (independent variables)
X = df.drop(columns=[target_column])

# Target (dependent variable)
y = df[target_column]

print("\nFeature Data (X):")
print(X.head())

print("\nTarget Data (y):")
print(y.head())

# -------------------------------
# 4. Import train_test_split
# -------------------------------
from sklearn.model_selection import train_test_split

# -------------------------------
# 5. Define split parameters
# -------------------------------
test_size_ratio = 0.2   # 20% test data
random_seed = 42        # ensures same split every time

# -------------------------------
# 6. Perform the split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,                      # features
    y,                      # target
    test_size=test_size_ratio,
    random_state=random_seed
)

# -------------------------------
# 7. Display split results
# -------------------------------
print("\n--- Split Results ---")
print("Total samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# -------------------------------
# 8. View sample data after split
# -------------------------------
print("\nSample X_train:")
print(X_train.head())

print("\nSample X_test:")
print(X_test.head())

print("\nSample y_train:")
print(y_train.head())

print("\nSample y_test:")
print(y_test.head())
