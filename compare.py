import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('employee_salary_data.csv')
num_cols = ['Experience', 'Age']
cat_cols = ['Education', 'Department']
y = data['Salary']

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def average_error(model, use_words):
    if use_words:
        X = data[num_cols + cat_cols]
        prep = ColumnTransformer([
            ('numbers', StandardScaler(), num_cols),
            ('words', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ])
    else:
        X = data[num_cols]
        prep = ColumnTransformer([('numbers', StandardScaler(), num_cols)])
    pipe = Pipeline([('prep', prep), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_absolute_error')
    return round(-scores.mean())

print("Linear Regression, numbers only :", average_error(LinearRegression(), False))
print("Linear Regression, numbers+words:", average_error(LinearRegression(), True))
print("Random Forest,     numbers only :", average_error(RandomForestRegressor(random_state=42), False))
print("Random Forest,     numbers+words:", average_error(RandomForestRegressor(random_state=42), True))