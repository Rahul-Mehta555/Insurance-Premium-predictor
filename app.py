
import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("insurance_model.pkl")



st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Insurance Premium Predictor")
st.write("Enter customer details to predict the insurance premium.")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

st.divider()

# Predict button
if st.button("Predict Premium", type="primary", use_container_width=True):

    # Prepare input in the same format as training data
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "sex": gender,
        "smoker": smoking_status,
        "region": region,
        "children": children
    }])

    # Prediction
    prediction = model.predict(input_data)

    # Display result
    st.success("Prediction Completed!")

    st.metric(
        label="Predicted Insurance Premium",
        value=f"₹ {prediction[0]:,.2f}"
    )
