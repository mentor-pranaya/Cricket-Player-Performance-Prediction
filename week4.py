import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Step 1 : Load engineered feature dataset
# ---------------------------------------

df = pd.read_csv("batsman_features.csv")

print(df.head())
print(df.shape)

# ---------------------------------------
# Step 2 : Check columns and data types
# ---------------------------------------

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

# ---------------------------------------
# Step 3 : Check missing values
# ---------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------
# Step 4 : Basic statistics
# ---------------------------------------

print("\nSummary statistics:")
print(df.describe())

# ---------------------------------------
# Step 5 : Distribution of target variable
# ---------------------------------------

plt.figure(figsize=(6,4))
sns.histplot(df['runs'], bins=30)
plt.title("Distribution of Runs (Target)")
plt.tight_layout()
plt.savefig("week4_runs_distribution.png")
plt.close()

# ---------------------------------------
# Step 6 : Distribution of main features
# ---------------------------------------

feature_cols = [
    'matches_played',
    'avg_runs_last_5',
    'avg_sr_last_5',
    'career_avg_runs'
]

for col in feature_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"week4_{col}_distribution.png")
    plt.close()

# ---------------------------------------
# Step 7 : Correlation analysis
# ---------------------------------------

corr = df[feature_cols + ['runs']].corr()

plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("week4_feature_correlation.png")
plt.close()

# ---------------------------------------
# Step 8 : Sort data (important for ML)
# ---------------------------------------

df = df.sort_values(['batter', 'season']).reset_index(drop=True)

# ---------------------------------------
# Step 9 : Train – Test split (time aware)
# ---------------------------------------

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

print("\nTrain size:", train_df.shape)
print("Test size :", test_df.shape)

# ---------------------------------------
# Step 10 : Save train & test files
# ---------------------------------------

train_df.to_csv("train_features.csv", index=False)
test_df.to_csv("test_features.csv", index=False)

print("\ntrain_features.csv and test_features.csv created")
