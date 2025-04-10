import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

#import Data Visualization library
import seaborn as sns
import matplotlib.pyplot as plt

#import Evaluate a Machine Learning Model library
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("youtube_dataset.csv")
print(df.head())

# 2.2 Perform Exploratory Data Analysis (EDA) and Visualize Key Insights
#
#   (a) Exploratory Data Analysis
#       - Compute basic stats (mean, median, std, etc.).
#       - Identify correlations, outliers, or data imbalances.
#       - Use pandas describe(), info(), corr() for an overview.

print("---------- 2.2 Perform Exploratory Data Analysis (EDA) and Visualize Key Insights ----------")
print("\n---------- Cleaned Data Summary: ----------")
print(df.describe())

print("\n---------- Data Structure (info()): ----------")
print(df.info())

print("\n---------- Correlation Analysis with corr() ----------")
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix (corr()):")
print(corr_matrix)

#   (b) Data Visualization
#       - Requirement: At least 3 different visualization techniques (histogram, 
#         scatter plot, box plot, heatmap, etc.).
#       - Tips: Use clear labels, titles, and legends. Let visuals drive your EDA narrative.

print("\n---------- box plot ----------")
plt.figure(figsize=(12,6))
sns.boxplot(data=df[["Total Subscribers", "Engagement Score", "AI Generated Content (%)"]])
plt.title("Outlier Detection (Numerical Features)")
plt.xticks(rotation=45)
plt.show()

print("\n---------- heatmap ----------")
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("\n---------- histogram ----------")
numerical_features = [
    "Total Subscribers", 
    "Engagement Score", 
    "AI Generated Content (%)", 
    "Avg Video Length (min)"
]

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], bins=20, kde=True, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()