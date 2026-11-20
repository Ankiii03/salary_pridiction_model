import joblib
import pandas as pd
import streamlit as st

model = joblib.load('salary_model.pkl')

# Read the data only to fill the dropdown lists
data = pd.read_csv('Salary_Data.csv').dropna()
data['Education Level'] = (
    data['Education Level'].str.lower().str.replace(' degree', '', regex=False).str.strip()
)

st.title("Salary Predictor")
st.caption("Random Forest trained on 1,783 cleaned records")

age = st.number_input("Age", 18, 70, 25)
exp = st.number_input("Years of Experience", 0, 45, 2)
gender = st.selectbox("Gender", sorted(data['Gender'].unique()))
edu = st.selectbox("Education Level", sorted(data['Education Level'].unique()))
job = st.selectbox("Job Title", sorted(data['Job Title'].unique()))

if st.button("Predict salary"):
    row = pd.DataFrame([{
        'Age': age,
        'Years of Experience': exp,
        'Gender': gender,
        'Education Level': edu,
        'Job Title': job,
    }])
    st.success(f"Estimated salary: {model.predict(row)[0]:,.0f}")