import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# ------------------------------------------------------------------------------
# 2. PROJECT TASKS IN DETAIL
# ------------------------------------------------------------------------------
#
# 2.1 Acquire, Clean, and Preprocess Data
#
#   (a) Data Acquisition
#       - Identify your data source: file-based (CSV, JSON), database, API, etc.
#       - Document how you obtained it. For example, if from an API, show the request.
print("---------- 2.1 Acquire, Clean, and Preprocess Data ----------")
df = pd.read_csv("youtube_dataset.csv")
print(df.info())
print(df.head())

#   (b) Data Cleaning
#       - Tasks: Handle missing values, remove duplicates, correct invalid entries.
#       - Python Tools: pandas methods (isnull, dropna, fillna, etc.).
#       - Tips: Always justify your decisions, e.g., why dropping vs. imputing missing values.
print("----------Missing values per column:\n----------")
missing_summary = df.isnull().sum()
print("Missing values per column:\n", missing_summary)

print("---------- Data Missing Values ----------")
df_isnull = df.isnull()
print("\nData After data Missing Values:\n", df_isnull)

print("---------- Dropping Missing Value ----------")
df_dropped = df.dropna()
print("\nData After Dropping Missing Values:\n", df_dropped)

print("---------- DFilling Missing Values----------")
df_filled = df.fillna(0)
print("\nData After Filling Missing Values:\n", df_filled)

#   (c) Data Preprocessing
#       - Requirement: Use at least 2 preprocessing techniques 
#         (scaling, encoding, feature engineering, etc.).
#       - Tips: Ensure numeric vs. categorical variables are appropriately transformed.

print("---------- Encoding Categorical Variables ----------")
# Convert boolean to binary
df["Neural Interface Compatible"] = df["Neural Interface Compatible"].astype(int)

print("---------- Feature Engineering ----------")
# Create new feature: Engagement per Subscriber
df["Engagement per Subscriber"] = df["Engagement Score"] / df["Total Subscribers"]

# Display preprocessed data
print("\nPreprocessed Data:")
print(df.head(3))