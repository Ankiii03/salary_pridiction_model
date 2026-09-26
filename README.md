# Salary Prediction Model 💼

A machine learning project that predicts employee salaries based on historical data, built during an AI/ML internship at IBM (via Edunet Foundation - IBM SkillsBuild, June–July 2025).

## Overview

This project processes and analyzes employee salary data to build a predictive model estimating salary based on relevant features such as experience, education, and job role. It compares multiple regression models and evaluates them using standard performance metrics.

## Dataset

- `employee_salary_data.csv` — employee records used for training and evaluation
- Processed and cleaned 100,000+ records as part of the pipeline, including handling missing values and encoding categorical variables.

## Approach

1. **Data Cleaning & Preparation** — handled missing values, encoded categorical features, and prepared the dataset for modeling.
2. **Model Training** — trained and compared two regression models:
   - Linear Regression
   - Random Forest Regressor
3. **Evaluation** — assessed model performance using:
   - RMSE (Root Mean Squared Error)
   - R² Score
4. **Prediction Interface** — built a simple UI for real-time salary prediction based on user input.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Notebook:** `salaryprediction.ipynb`

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Ankiii03/salary_priction_model.git
cd salary_priction_model
