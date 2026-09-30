import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write("Enter the student details below to predict the Math Score.")

gender = st.selectbox("Gender", ["female", "male"])

race = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

parent = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox(
    "Lunch",
    ["standard", "free/reduced"]
)

prep = st.selectbox(
    "Test Preparation Course",
    ["none", "completed"]
)

reading = st.slider("Reading Score", 0, 100, 70)

writing = st.slider("Writing Score", 0, 100, 70)

if st.button("Predict Math Score"):

    input_data = pd.DataFrame({
        "gender":[gender],
        "race/ethnicity":[race],
        "parental level of education":[parent],
        "lunch":[lunch],
        "test preparation course":[prep],
        "reading score":[reading],
        "writing score":[writing]
    })

    prediction = model.predict(input_data)

    st.success(f"🎯 Predicted Math Score: {prediction[0]:.2f}")