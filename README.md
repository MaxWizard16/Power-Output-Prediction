# ⚡ Power Consumption Forecasting using Machine Learning

## 📌 Overview

This project focuses on predicting future power consumption using historical time-series data. It combines data analysis, feature engineering, and machine learning to forecast power values **15 minutes ahead**.

The goal is to understand patterns in power usage and build a model that can make reliable short-term predictions.

---

## 🎯 Objective

* Analyze real-time power consumption data
* Identify trends and patterns in time-series data
* Build a machine learning model for short-term forecasting

---

## 📊 Dataset

The dataset contains timestamped power consumption values.

**Key columns:**

* `Datetime` → Timestamp of observation
* `Power` → Power consumption value

---

## 🧠 Project Workflow

### 1. Data Preprocessing

* Converted `Datetime` column into proper datetime format
* Sorted dataset chronologically
* Checked for missing values and inconsistencies

---

### 2. Exploratory Data Analysis (EDA)

* Missing value visualization using heatmaps
* Distribution analysis using KDE plots
* Outlier detection using boxplots
* Time-series visualization of power trends

---

### 3. Feature Engineering

* Created rolling mean (60-step smoothing)
* Generated lag features:

  * 1 step
  * 5 steps
  * 15 steps
  * 30 steps
  * 60 steps
* Performed correlation analysis to identify important predictors

---

### 4. Time-Series Analysis

* Conducted stationarity check using Augmented Dickey-Fuller (ADF) test

---

### 5. Model Development

* Target variable: Power shifted 15 steps ahead
* Model used: **Random Forest Regressor**
* Train-test split performed without shuffling to preserve time order

---

### 6. Model Evaluation

The model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

---

### 7. Visualization

* Actual vs Predicted power values
* Feature importance analysis

---

## 📈 Results

The model successfully captures short-term patterns in power consumption and provides reasonably accurate predictions for future values.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels

---

## 📌 Key Learnings

* Working with real-world time-series data
* Importance of feature engineering (lags & rolling features)
* Avoiding data leakage in time-based problems
* Evaluating regression models effectively

---

## 🚀 Future Improvements

* Implement advanced models (XGBoost, LSTM, GRU)
* Hyperparameter tuning
* Incorporate external features (weather, demand trends)
* Deploy as a real-time prediction system

---

## 👨‍💻 Author

Tanish Bansal

---

## ⭐ If you found this useful

Give this repo a star ⭐ and feel free to fork or contribute!
