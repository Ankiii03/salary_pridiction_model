import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- 1. Load and clean ----------
data = pd.read_csv('Salary_Data.csv')
print("Raw rows:", len(data))

data = data.dropna()                       # remove rows with missing values
data = data.drop_duplicates()              # remove exact repeat rows

# Fix inconsistent education labels ("Bachelor's Degree" -> "bachelor's", etc.)
data['Education Level'] = (
    data['Education Level'].str.lower().str.replace(' degree', '', regex=False).str.strip()
)

before = len(data)
data = data[data['Salary'] >= 5000]        # tiny salaries look like typos
print("Removed", before - len(data), "rows with salary under 5000")
print("Clean unique rows (N):", len(data))

# ---------- 2. Inputs and answer ----------
num_cols = ['Age', 'Years of Experience']
cat_cols = ['Gender', 'Education Level', 'Job Title']
y = data['Salary']

# ---------- 3. Compare 4 setups with 5-fold cross-validation ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def make_pipeline(model, use_words):
    steps = [('numbers', StandardScaler(), num_cols)]
    if use_words:
        steps.append(('words', OneHotEncoder(handle_unknown='ignore'), cat_cols))
    return Pipeline([('prep', ColumnTransformer(steps)), ('model', model)])

def score(model, use_words):
    cols = num_cols + (cat_cols if use_words else [])
    pipe = make_pipeline(model, use_words)
    mae = -cross_val_score(pipe, data[cols], y, cv=cv, scoring='neg_mean_absolute_error').mean()
    r2 = cross_val_score(pipe, data[cols], y, cv=cv, scoring='r2').mean()
    return round(mae), round(r2, 3)

rf = lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)

print("\n(average error, R2)")
print("Linear Regression, numbers only :", score(LinearRegression(), False))
print("Linear Regression, numbers+words:", score(LinearRegression(), True))
print("Random Forest,     numbers only :", score(rf(), False))
print("Random Forest,     numbers+words:", score(rf(), True))

# ---------- 4. Final test of the best setup on hidden data ----------
X = data[num_cols + cat_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final = make_pipeline(rf(), True)
final.fit(X_train, y_train)
pred = final.predict(X_test)

print("\nFinal Random Forest on hidden 20%:")
print("Test rows :", len(X_test))
print("R2        :", round(r2_score(y_test, pred), 3))
print("RMSE      :", round(np.sqrt(mean_squared_error(y_test, pred))))
print("MAE       :", round(mean_absolute_error(y_test, pred)))

joblib.dump(final, 'salary_model.pkl', compress=3)
print("Saved salary_model.pkl")