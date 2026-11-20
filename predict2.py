import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

data = pd.read_csv('employee_salary_data.csv')

# Number columns and text columns
num_cols = ['Experience', 'Age']
cat_cols = ['Education', 'Department']

X = data[num_cols + cat_cols]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step A: prepare the data (scale numbers, one-hot the words)
prep = ColumnTransformer([
    ('numbers', StandardScaler(), num_cols),
    ('words', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

# Step B: bundle preparation + model into one pipeline
model = Pipeline([
    ('prep', prep),
    ('model', LinearRegression()),
])

model.fit(X_train, y_train)

guess = model.predict(X_test)
print("Real salaries :", list(y_test))
print("Model guesses :", [round(g) for g in guess])

# New person: pass ALL four columns, in plain words/numbers
new_person = pd.DataFrame([{
    'Experience': 4, 'Age': 27,
    'Education': 'Master', 'Department': 'IT'
}])
print("New person's salary guess:", round(model.predict(new_person)[0]))
