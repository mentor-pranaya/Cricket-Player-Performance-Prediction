🏏 AI-Based Cricket Player Performance Prediction System

An end-to-end Machine Learning project that predicts batsman runs and bowler wickets using historical IPL data.

The system includes:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Hyperparameter Tuning
- Model Evaluation
- Explainable AI (SHAP)
- Interactive Streamlit Dashboard

---

📌 Project Overview

Cricket performance prediction is complex and data-intensive.  
This project builds a fully automated, data-driven prediction system that:

- Predicts runs scored by a batsman
- Predicts wickets taken by a bowler
- Provides interactive visualization via Streamlit

---

📊 Dataset Information

Source: Kaggle – IPL Dataset (2008–2025)

| Dataset | Rows | Columns |
|----------|-------|----------|
| ball_by_ball_data | 278,205 | 30 |
| ipl_matches_data | 1,169 | 23 |
| players_data_updated | 772 | 6 |
| team_aliases | 46 | 3 |
| teams_data | 16 | 2 |

---

🧹 Data Cleaning

- Removed duplicate records
- Handled missing values
- Standardized team and player names
- Converted date & numeric columns
- Merged datasets properly
- Ensured no data leakage

---

📈 Exploratory Data Analysis (EDA)

Performed detailed EDA to understand patterns:

- Runs distribution across seasons
- Wickets distribution trends
- Team-wise average scoring
- Player consistency analysis
- Venue impact on performance
- Strike rate and economy analysis

EDA helped in identifying important predictive features.

---

⚙️ Processing & Feature Engineering

Created meaningful cricket-based features such as:
 For Batsman:
- `form_runs_last_10`
- Career average
- Strike rate
- Boundary percentage
- Venue average runs
- Opponent average runs
- Team average runs

For Bowler:
- form_wickets_last_10`
- Bowling economy rate
- Career wickets
- Opponent wicket rate
- Venue wicket frequency
- Team bowling average

These features significantly improved model performance.

---

 Machine Learning Models

Three regression models were trained for both tasks:

- Random Forest
- XGBoost
- LightGBM

Train-Test Split
- 80% Training Data
- 20% Testing Data
- No data leakage ensured

---

 📊 Model Evaluation Metrics

Used:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² Score (Coefficient of Determination)

---

 ✅ Final Model Performance

🏏 Batsman Model (XGBoost – Best Model)

- MAE:9.045
- RMSE:12.169
- R²:0.705

---

🎯 Bowler Model (Random Forest – Best Model)

- MAE:0.829
- RMSE:1.055
- R²:0.045

Predicting wickets is more challenging due to high randomness in bowling outcomes.

---

Explainable AI (SHAP)

SHAP is used to:

- Show feature importance
- Explain individual predictions
- Improve transparency
- Build trust in the system

It displays how each feature contributes to predicted runs or wickets.

---

🖥️ Streamlit Dashboard Features

- Player & team selection
- Predicted runs & wickets
- Last 10 match trend graphs
- SHAP explanation graph
- Team summary mode