**Cricket Player Performance Prediction**

**Project Overview**

I developed this project to predict the performance of cricket players (specifically within the IPL) by leveraging historical ball-by-ball data and match summaries. The goal is to provide data-driven insights into player form, venue impact, and expected performance in upcoming matches.

**Key Features**

• Data Pipeline: Automated cleaning and preprocessing of raw IPL datasets.

• Feature Engineering: Advanced stats including rolling averages (form), venue-specific performance, and opponent-specific matchups.

• Machine Learning: Predictive models built using XGBoost, Random Forest, and LightGBM.

• Interpretability: Model decisions explained using SHAP values to identify key performance drivers.

• Interactive Dashboard: A Streamlit application that allows users to input match parameters and receive real-time predictions.

**Repository Structure**

The project is organized into the following directories:

• data/processed/: Contains the final feature-engineered dataset.csv.

• notebooks/:

&nbsp;   ◦ 01\_EDA.ipynb: Exploratory data analysis and initial findings.

&nbsp;   ◦ 02\_FeatureEngineering.ipynb: Logic for feature creation and preprocessing.

&nbsp;   ◦ 03\_ModelTraining.ipynb: Model training, tuning, and evaluation logs.

• src/: Modular Python scripts including data\_cleaning.py and app.py(The main entry point for the interactive dashboard).

• dataset\_details.txt: Documentation regarding the origin and schema of the datasets used.

**Getting Started**

1\. Installation

Clone the repository and install the necessary dependencies:

git clone https://github.com/mentor-pranaya/Cricket-Player-Performance-Prediction.git

cd Cricket-Player-Performance-Prediction

pip install -r requirements.txt

2\. Running the Dashboard

To launch the prediction interface, run:

streamlit run app.py

**Development History**

This project followed a structured 8-week development lifecycle:

• Weeks 1–2: Data acquisition and exploratory analysis.

• Weeks 3–4: Feature engineering and creation of training labels.

• Weeks 5–6: Model development, hyperparameter tuning, and evaluation.

• Weeks 7–8: Dashboard integration and deployment.

