# 🏏 Cricket Player Performance Prediction (IPL)

An ML-driven system that predicts individual player performance in IPL T20 matches — runs for batsmen, wickets for bowlers — using historical ball-by-ball data from 2008 to 2025.

## Features

- **Runs Prediction** for batsmen (Random Forest regressor)
- **Wickets Prediction** for bowlers (Random Forest regressor)
- **SHAP Feature Importance** showing which factors drive each prediction
- **Form Trend Charts** visualizing recent match-by-match performance
- **Career Analytics** with venue and opponent breakdowns
- **Interactive Dashboard** built with Streamlit

## Tech Stack

| Component | Tools |
|---|---|
| Language | Python 3.x |
| Data Handling | pandas, numpy |
| Visualization | plotly, matplotlib, seaborn |
| Machine Learning | scikit-learn, xgboost |
| Model Explainability | SHAP |
| Model Serialization | joblib |
| Dashboard | Streamlit |
| Dataset | [Kaggle IPL Ball-by-Ball (2008–2025)](https://www.kaggle.com/) |

## Installation

```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib shap plotly matplotlib seaborn
```

## Running the App

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## Project Structure

```
├── app.py                          # Streamlit dashboard (main entry point)
├── prepare_production_data.py      # Generates lookup tables for predictions
├── batsman-model-training-milestone3.ipynb   # Batsman model training notebook
├── bowler-model-training-milestone3.ipynb    # Bowler model training notebook
├── batsman_features_final.csv      # Feature-engineered batsman dataset
├── bowler_features_final.csv       # Feature-engineered bowler dataset
├── batsman_pipeline.pkl            # Trained batsman RF pipeline (preprocessor + model)
├── bowler_pipeline.pkl             # Trained bowler RF pipeline (preprocessor + model)
├── batsman_xgb.pkl                 # Trained batsman XGBoost model
├── bowler_xgb.pkl                  # Trained bowler XGBoost model
├── batsman_preprocessor.pkl        # Standalone preprocessor for XGBoost
├── bowler_preprocessor.pkl         # Standalone preprocessor for XGBoost
├── prod_batsman_base.csv           # Player base stats (smoothed recent form)
├── prod_bowler_base.csv            # Player base stats (smoothed recent form)
├── prod_bat_vs_opp.csv             # Batsman vs opponent lookup
├── prod_bowl_vs_opp.csv            # Bowler vs opponent lookup
├── prod_bat_at_venue.csv           # Batsman at venue lookup
├── prod_bowl_at_venue.csv          # Bowler at venue lookup
├── dashboard_batsman_data.csv      # Dashboard analytics data (batsman)
├── dashboard_bowler_data.csv       # Dashboard analytics data (bowler)
└── README.md                       # This file
```

## How It Works

1. **Data Preprocessing**: Ball-by-ball IPL data is aggregated to player-match level
2. **Feature Engineering**: Rolling averages (form), venue stats, opponent-specific stats, career metrics
3. **Model Training**: Random Forest and XGBoost regressors with time-series train/val/test split
4. **Prediction**: User selects player + opponent + venue → model predicts runs or wickets
5. **Explainability**: SHAP values show which features drove the prediction

## Model Performance

### Batsman (Predicting Runs)
| Model | MAE |
|---|---|
| Mean Baseline | 18.16 |
| Rolling Avg Baseline | 17.56 |
| **Random Forest** | **10.80** |
| XGBoost | 11.06 |

### Bowler (Predicting Wickets)
| Model | MAE |
|---|---|
| Mean Baseline | 0.816 |
| Rolling Avg Baseline | 0.879 |
| **Random Forest** | **0.846** |
| XGBoost | 0.880 |

## Dataset

IPL ball-by-ball dataset from Kaggle covering seasons 2008–2025. Includes match metadata, player statistics, and ball-level events.
