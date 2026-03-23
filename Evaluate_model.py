from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to evaluate the model using actual data and model predictions
def evaluate_model_performance(y_test, predictions):
    print("Evaluating Model Performance...")
    
    # Calculating Errors
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Printing Results
    print("--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2) Score: {r2}")
    
    return mae, mse, r2

# Note for team: Call this function after model.predict() in the main execution
print("Evaluation module loaded successfully.")
