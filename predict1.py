import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('employee_salary_data.csv')

# 1) Pick inputs (X) and the answer (y)
X = data[['Experience', 'Age']]
y = data['Salary']

# 2) Split: 80% to learn from, 20% hidden to test later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Create the model and let it learn
model = LinearRegression()
model.fit(X_train, y_train)

# 4) Test it on the people it has never seen
guess = model.predict(X_test)
print("Real salaries :", list(y_test))
print("Model guesses :", [round(g) for g in guess])

# 5) Predict for a brand new person: 4 years experience, age 27
new_person = pd.DataFrame([[4, 27]], columns=['Experience', 'Age'])
print("New person's salary guess:", round(model.predict(new_person)[0]))