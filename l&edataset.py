import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("insurance_data_linear.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nColumns:")
print(data.columns)

print("\nShape of dataset:")
print(data.shape)

plt.scatter(data["age"], data["charges"])
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Age vs Charges")
plt.show()