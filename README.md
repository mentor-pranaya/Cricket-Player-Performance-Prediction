🏏 Cricket Player Performance Prediction

📌 Project Overview

This project develops a machine learning–based predictive system to estimate cricket player performance in upcoming matches.

The system predicts:

- Expected Runs (Batting Model)

- Expected Wickets (Bowling Model)

Using historical IPL ball-by-ball data, the project implements:

Advanced feature engineering

Ensemble regression modeling

Model comparison & evaluation

Explainable AI (SHAP)

Interactive Streamlit dashboard deployment


🎯 Objective

To build a context-aware predictive framework that estimates a player’s match performance based on:

Career statistics

Recent form

Opponent strength

Venue impact

Experience level

The goal is to move beyond descriptive averages and create a predictive analytics system.


📊 Dataset

Source: IPL Ball-by-Ball Dataset

Core Files Used

matches.csv

deliveries.csv

After preprocessing:

matches_cleaned.csv

deliveries_cleaned.csv

final_dataset.csv

bowler_dataset.csv


🛠 Feature Engineering

Batting Features

Career average runs

Recent form (rolling average)

Venue-specific average

Opponent-specific average

Match experience

Consistency metrics

Bowling Features

Career wicket average

Recent wickets form

Opponent wicket impact

Feature engineering significantly improved predictive accuracy.


🤖 Modeling Approach

Separate regression models were trained for:

🟢 Runs Prediction

Model: xgb_model.joblib

🔵 Wickets Prediction

Model: wicket_model.joblib

Other models evaluated:

Random Forest

LightGBM (lgbm_model.joblib)

Baseline (Rolling Average)

Final selected model: XGBoost


📈 Model Evaluation

Evaluation metrics used:

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

R² Score
 
Model	         MAE	 RMSE	  R²
Baseline	    16.50	22.25 -0.004
Random Forest	16.22	21.26	0.083
XGBoost	      15.67	20.81	0.121
LightGBM	    15.67	20.93	0.111

XGBoost achieved the best overall performance.


📊 Interactive Dashboard

File: streamlit_app1.py

Built using Streamlit, the dashboard includes:

🔹 Input Controls

Player selection

Opponent selection

Venue selection

🔹 Output Section

Predicted Runs

Predicted Wickets

Recent performance graph

Residual distribution

SHAP feature importance

The dashboard integrates prediction, visualization, and interpretability.


🧠 Model Interpretability

SHAP (SHapley Additive explanations) is used to:

Explain model predictions

Identify key influencing features

Improve transparency of ensemble models

🏗 Project Structure

CPPREDICTION/
│
├── DATA1/                     # Raw and processed datasets
│   ├── bowler_dataset.csv
│   ├── deliveries.csv
│   ├── deliveries_cleaned.csv
│   ├── final_dataset.csv
│   ├── matches.csv
│   └── matches_cleaned.csv
│
├── models1/                   # Trained ML models and pipelines
│   ├── feature_pipeline.pkl
│   ├── lgbm_model.joblib
│   ├── model.joblib
│   ├── wicket_model.joblib
│   └── xgb_model.joblib
│
├── NOTEBOOKS/                 # Jupyter notebooks for development
│   ├── 1.EDA.ipynb
│   ├── 2.featureengg.ipynb
│   └── 3.modeltraining.ipynb
│
├── data_cleaning1.py          # Data preprocessing script
├── streamlit_app1.py          # Streamlit dashboard application
└── README.md                  # Project documentation
