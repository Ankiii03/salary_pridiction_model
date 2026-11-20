# Salary Prediction Model

Predicts employee salary from age, years of experience, gender, education level and job title,
and serves predictions through a small Streamlit web app.

## Results (Salary_Data.csv)
- Raw records: 6,704 → 1,783 unique records after cleaning (missing values, duplicates,
  inconsistent education labels, salaries under 5,000 removed)
- Random Forest on a hidden 20% test set (357 rows): R² = 0.893, RMSE = 17,050, MAE = 11,328

| Model (5-fold cross-validation) | Average error (MAE) | R² |
|---|---|---|
| Linear Regression, numbers only | 23,316 | 0.669 |
| Linear Regression, numbers + words | 15,070 | 0.841 |
| Random Forest, numbers only | 18,011 | 0.777 |
| Random Forest, numbers + words | 11,000 | 0.899 |

## How to run
```
pip install pandas scikit-learn joblib streamlit
python big.py
streamlit run app.py
```

## Files
- `big.py` – cleaning, model comparison, saves `salary_model.pkl`
- `app.py` – Streamlit web app
- `salarypridiction.ipynb`, `employee_salary_data.csv` – first small prototype (10 rows)

![App screenshot](streamlit.png)
