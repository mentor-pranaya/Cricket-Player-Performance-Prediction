# 🏏 IPL Player Performance Prediction using Machine Learning

## 📌 Overview

This project is an end-to-end sports analytics system that predicts IPL player performance using Machine Learning.
It analyzes historical IPL ball-by-ball data, creates performance features, trains predictive models, and provides an interactive dashboard to visualize insights.

The system predicts the number of runs a batsman is likely to score in an upcoming match and explains the reasoning behind the prediction using Explainable AI.

---

## 🎯 Objectives

* Predict IPL batsman performance using historical data
* Analyze player form, career statistics, and match context
* Provide explainable predictions using SHAP
* Build an interactive analytics dashboard

---

## ⚙️ Features

* ✅ Player runs prediction using XGBoost
* ✅ Feature engineering (career average, recent form, strike rate, boundaries)
* ✅ Player comparison analytics
* ✅ Recent performance visualization
* ✅ Explainable AI (SHAP feature importance)
* ✅ Prediction logging
* ✅ Interactive Streamlit dashboard

---

## 🗂 Dataset

The project uses IPL ball-by-ball and match datasets containing:

* Match details
* Player statistics
* Batting and bowling performance
* Venue and opponent information

---

## 🧠 Machine Learning Approach

1. Data preprocessing and cleaning
2. Feature engineering (career stats, recent form, venue performance)
3. Model training using **XGBoost Regressor**
4. Model evaluation
5. Explainability using **SHAP**

---

## 🖥 Dashboard

The Streamlit dashboard allows users to:

* Select players
* Compare players
* View predicted runs
* See recent form graphs
* Understand prediction reasoning via SHAP

---

## 🏗 Project Structure

```
ipl_prediction_project/
│── data/
│── models/
│── preprocessing.py
│── feature_engineering.py
│── train_model.py
│── app.py
│── requirements.txt
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone repository

```
git clone <your-repo-link>
cd ipl_prediction_project
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run preprocessing & training

```
python preprocessing.py
python feature_engineering.py
python train_model.py
```

### 4️⃣ Run dashboard

```
streamlit run app.py
```

---

## 🛠 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib

---

## 📊 Applications

* Fantasy cricket decision making
* Sports analytics
* Player performance analysis
* Match strategy insights

---

## ⚠️ Limitations

* Predictions depend on historical data quality
* Does not consider real-time factors (injuries, pitch conditions, weather)

---

## 🔮 Future Scope

* Match winner prediction
* Bowler performance prediction
* Dream11 team recommendation
* Real-time API integration
* Deep learning models

---

## 👨‍💻 Author

Ausaf Ahmed Ansari

---

## ⭐ Acknowledgement

IPL datasets from public cricket analytics sources and Kaggle.

---

## 📜 License

This project is for educational and research purposes.
