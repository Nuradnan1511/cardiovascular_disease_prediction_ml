# Cardiovascular Disease Prediction Using Machine Learning

## Project Overview

This project develops a machine learning system to predict the presence of cardiovascular disease based on patient health and lifestyle information.

Multiple supervised machine learning algorithms were trained and evaluated, and hyperparameter tuning was performed to improve the performance of the selected model.

🚀 **[Live Demo](https://cardiovasculardiseasepredictionml-nqdhgzxkmt5gense3scmbf.streamlit.app/)**
---
---

## Objectives

- Perform exploratory data analysis on cardiovascular disease data
- Clean and preprocess the dataset
- Engineer meaningful features
- Train multiple machine learning models
- Compare model performance
- Perform hyperparameter tuning
- Select the best-performing model
- Develop an interactive Streamlit application
- Deploy the project as a reproducible machine learning application

---

## Dataset

The project uses the Cardiovascular Disease dataset containing patient demographic, medical, and lifestyle information.

The dataset includes variables such as:

- Age
- Gender
- Height
- Weight
- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol
- Glucose
- Smoking
- Alcohol consumption
- Physical activity

The target variable indicates whether cardiovascular disease is present.
dataset link: https://www.kaggle.com/datasets/sulianova/cardiovasc

---

## Feature Engineering

Several additional features were created:

- Age in years
- BMI
- Blood pressure difference
- Blood pressure ratio

### BMI

BMI was calculated using:

BMI = Weight / Height²

where height is converted from centimeters to meters.

### Blood Pressure Difference

BP Difference = Systolic BP - Diastolic BP

### Blood Pressure Ratio

BP Ratio = Systolic BP / Diastolic BP

---

## Machine Learning Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine
6. Gradient Boosting

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 73.03% | 74.75% | 69.11% | 71.82% | 79.78% |
| Logistic Regression | 72.42% | 74.84% | 67.09% | 70.75% | 78.80% |
| SVM | 72.29% | 75.13% | 66.18% | 70.37% | 78.76% |
| Random Forest | 70.62% | 70.77% | 69.72% | 70.24% | 76.42% |
| KNN | 69.45% | 69.75% | 68.11% | 68.92% | 74.36% |
| Decision Tree | 62.05% | 61.71% | 62.44% | 62.07% | 62.05% |

---

## Final Model

Gradient Boosting was selected as the final model after model comparison and hyperparameter tuning.

The final model uses:

- `n_estimators = 200`
- `learning_rate = 0.05`
- `min_samples_split = 5`
- `min_samples_leaf = 4`
- `subsample = 0.9`
- `random_state = 42`

---

## Streamlit Application

The project includes an interactive Streamlit application.

Users can enter:

- Age
- Gender
- Height
- Weight
- Blood pressure
- Cholesterol
- Glucose
- Smoking status
- Alcohol consumption
- Physical activity

The application automatically calculates engineered features and generates a prediction using the trained Gradient Boosting model.

---


## Project Pipeline


Kaggle Dataset
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train-Test Split & Scaling
      ↓
Train 6 ML Classification Models
      ↓
Model Evaluation & Comparison
      ↓
Hyperparameter Tuning
      ↓
Select Best Model
      ↓
Save Trained Model
      ↓
Build Streamlit Application
      ↓
Deploy to Streamlit Cloud
