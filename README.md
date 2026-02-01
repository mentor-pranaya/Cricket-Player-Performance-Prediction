# Week 1: Exploratory Data Analysis (EDA)

## Project Title
Cricket Player Performance Prediction

## Intern Name
Himanethri Kanamarla Pudi

## Objective
The goal of Week 1 is to understand the dataset by performing:
- Data loading
- Data cleaning
- Data preprocessing
- Exploratory Data Analysis (EDA)

This helps in identifying patterns, trends, and issues in the data before building machine learning models.

---

## Dataset Description
The dataset used is **IPL Men’s Ball-by-Ball Data (up to 2024)**.

**Files used:**
- `deliveries_updated_mens_ipl_upto_2024.csv`

**Key columns:**
- `batsman`, `bowler`
- `batting_team`, `bowling_team`
- `batsman_runs`, `extras`
- `dismissal_kind`
- `date`

---

## Tools & Libraries Used
- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Steps Performed

### 1. Data Loading
- Loaded CSV file using Pandas
- Verified file path and structure

### 2. Data Understanding
- Checked shape of dataset
- Viewed column names
- Inspected first few rows

### 3. Data Cleaning
- Identified missing values
- Handled NaN values logically
- Verified data types

### 4. Exploratory Data Analysis (EDA)
- Distribution of runs
- Team-wise analysis
- Player performance insights
- Match and innings-based trends

---

## Key Insights (Week 1)
- Dataset contains ball-by-ball match data
- Some columns have missing values which need handling
- Strong scope for player performance modeling

---

## Conclusion
Week 1 successfully focused on understanding and exploring the dataset.
This forms a strong foundation for feature engineering and model building in upcoming weeks.

---

---

## Week 2: Feature Engineering

### Objective
The goal of Week 2 is to transform raw cricket match data into meaningful numerical features
that can be used for training machine learning models.

---

### What was done in Week 2

1. Created player-level performance features from ball-by-ball data
2. Aggregated batting statistics such as:
   - Total runs scored
   - Total balls faced
   - Number of matches played
3. Engineered important performance metrics like:
   - Batting average
   - Strike rate
4. Handled missing values created during aggregation
5. Prepared a clean, model-ready dataset

---

### Why Feature Engineering is Important

Raw data cannot be directly used by machine learning models.
Feature engineering helps convert cricket match events into numerical signals
that represent a player's performance and consistency.

---

### Files Used

- `notebooks/02_feature_engineering.ipynb`
- `notebooks/01_EDA_and_02_Feature_Engineering.ipynb`

---

### Output of Week 2

A clean, structured dataset with engineered features that will be used
for model training in Week 3.

