import pandas as pd

data = pd.read_csv('employee_salary_data.csv')

print(data.head())
print("Rows:", len(data))
print(data.columns)