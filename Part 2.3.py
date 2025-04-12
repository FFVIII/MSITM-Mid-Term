import pandas as pd

#import Data Visualization library
import seaborn as sns
import matplotlib.pyplot as plt

#import Evaluate a Machine Learning Model library
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 2.3 Build and Evaluate a Machine Learning Model
#
#   (a) Model Building
#       - Requirement: At least 2 different ML algorithms 
#         (e.g., Logistic Regression, Random Forest, Linear Regression, etc.).
#       - Tips: Match the algorithm type to your target variable 
#         (classification vs. regression).

# Load a dataset
df = pd.read_csv("youtube_dataset.csv")

print("---------- Linear Regression ----------")
X_reg = df[['Total Videos']]  
y_reg = df['Avg Video Length (min)']  

# Splitting data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

# Predict and evaluate
y_pred_reg = lin_reg.predict(X_test_reg)
print(f"R^2 Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")


print("---------- Logistic Regression (Classification) ----------")
# Features
X_clf = df[['Total Videos']] 
y_clf = df['Neural Interface Compatible']  

# Splitting data into training and testing sets
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Train model
log_reg = LogisticRegression()
log_reg.fit(X_train_clf_scaled, y_train_clf)

# Predict and evaluate
y_pred_clf = log_reg.predict(X_test_clf_scaled)
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_clf, y_pred_clf)}")
print(f"Classification Report:\n{classification_report(y_test_clf, y_pred_clf)}")

# Output test data (optional)
print("Test Data Set (Features for Linear Regression):")
print(X_test_reg)


#   (b) Model Evaluation
#       - Requirement: At least 2 different evaluation metrics 
#         (accuracy, precision/recall, F1, RMSE, MAE, etc.).
#       - Tips: Present numeric results and interpret them in plain English. 
#         Consider basic hyperparameter tuning.

print("---------- Evaluating a Regression Model ----------")
# Sample data (replace with your own)
X_train_reg = [[10], [20], [30]]  # Total Videos
y_train_reg = [5, 10, 15]         # Avg Video Length (min)
X_test_reg = [[25], [35]]         # Test data
y_test_reg = [12, 17]             # True values

# Train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg)  # Predictions: e.g., [12.5, 17.5]

# Calculate metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")

print("---------- Evaluating a Classification Model ----------")
# Sample data (replace with your own)
X_train_clf = [[10], [20], [30]]  # Total Videos
y_train_clf = [0, 1, 1]           # Compatible (0 = False, 1 = True)
X_test_clf = [[15], [25]]         # Test data
y_test_clf = [0, 1]               # True labels

# Train the model
log_reg = LogisticRegression()
log_reg.fit(X_train_clf, y_train_clf)
y_pred_clf = log_reg.predict(X_test_clf)  # Predictions: e.g., [0, 1]

# Calculate metrics
accuracy = accuracy_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")


