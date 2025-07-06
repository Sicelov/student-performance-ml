# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and features list
model = joblib.load("student_performance_model.pkl")
model_features = joblib.load("model_features.pkl")

# Display Feature Importance (model coefficients)
coefficients = pd.DataFrame(
    model.coef_, model_features, columns=["Coefficient"])
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

st.subheader("📊 Feature Importance (Model Coefficients)")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(coefficients.index, coefficients["Coefficient"], color="Darkblue")
ax.set_xlabel("Coefficient Value (Impact on Predicted Score)")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance in Student Score Prediction")
st.pyplot(fig)

st.title("🎓 Student Performance Predictor")

# User inputs for categorical features
gender = st.selectbox("Gender", ["Female", "Male"])
race = st.selectbox(
    "Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_education = st.selectbox("Parental Level of Education", [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
])
lunch = st.selectbox("Lunch", ["free/reduced", "standard"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Create input DataFrame with one row
input_dict = {
    'test preparation course': [1 if test_prep == "completed" else 0],
}

# Initialize all features to zero (including one-hot columns)
for feature in model_features:
    input_dict[feature] = [0]

# Fill in known features (note one-hot encoding format)
# Set binary feature
input_dict['test preparation course'] = [1 if test_prep == "completed" else 0]

# Gender one-hot: 'gender_male'
if gender == "Male":
    input_dict['gender_male'] = [1]

# Race/ethnicity one-hot: drop_first=True means no column for group A
if race != "group A":
    col_name = f"race/ethnicity_{race}"
    if col_name in input_dict:
        input_dict[col_name] = [1]

# Parental education
if parent_education != "some high school":
    col_name = f"parental level of education_{parent_education}"
    if col_name in input_dict:
        input_dict[col_name] = [1]

# Lunch
if lunch == "standard":
    input_dict['lunch_standard'] = [1]

# Convert to DataFrame with columns ordered correctly
input_df = pd.DataFrame(input_dict)[model_features]

if st.button("Predict Average Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Average Score: {prediction:.2f}")
