# ==========================================
# CARDIOVASCULAR DISEASE PREDICTION APP
# ==========================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/cardiovascular_disease_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


# ==========================================
# TITLE
# ==========================================

st.title(
    "❤️ Cardiovascular Disease Prediction"
)

st.write(
    "Machine Learning-Based Cardiovascular "
    "Disease Prediction"
)

st.markdown(
    """
    Enter the patient's information below to
    generate a prediction using the trained
    machine learning model.
    """
)


# ==========================================
# PATIENT INFORMATION
# ==========================================

st.header("Patient Information")


# Age
age_years = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)


# Gender
gender_option = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

if gender_option == "Male":
    gender = 1
else:
    gender = 2


# Height
height = st.number_input(
    "Height (cm)",
    min_value=50.0,
    max_value=250.0,
    value=170.0,
    step=1.0
)


# Weight
weight = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=300.0,
    value=70.0,
    step=1.0
)


# ==========================================
# BLOOD PRESSURE
# ==========================================

st.header("Blood Pressure")

ap_hi = st.number_input(
    "Systolic Blood Pressure",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)

ap_lo = st.number_input(
    "Diastolic Blood Pressure",
    min_value=30,
    max_value=150,
    value=80,
    step=1
)


# ==========================================
# HEALTH INFORMATION
# ==========================================

st.header("Health Information")


# Cholesterol
cholesterol_option = st.selectbox(
    "Cholesterol Level",
    [
        "Normal",
        "Above Normal",
        "Well Above Normal"
    ]
)

if cholesterol_option == "Normal":
    cholesterol = 1

elif cholesterol_option == "Above Normal":
    cholesterol = 2

else:
    cholesterol = 3


# Glucose
glucose_option = st.selectbox(
    "Glucose Level",
    [
        "Normal",
        "Above Normal",
        "Well Above Normal"
    ]
)

if glucose_option == "Normal":
    gluc = 1

elif glucose_option == "Above Normal":
    gluc = 2

else:
    gluc = 3


# Smoking
smoke_option = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

if smoke_option == "Yes":
    smoke = 1
else:
    smoke = 0


# Alcohol
alcohol_option = st.selectbox(
    "Alcohol Consumption",
    ["No", "Yes"]
)

if alcohol_option == "Yes":
    alco = 1
else:
    alco = 0


# Physical Activity
activity_option = st.selectbox(
    "Physical Activity",
    ["No", "Yes"]
)

if activity_option == "Yes":
    active = 1
else:
    active = 0


# ==========================================
# CALCULATED FEATURES
# ==========================================

bmi = weight / ((height / 100) ** 2)

bp_difference = ap_hi - ap_lo

bp_ratio = ap_hi / ap_lo


st.header("Calculated Features")

st.write(
    f"**BMI:** {bmi:.2f}"
)

st.write(
    f"**Blood Pressure Difference:** "
    f"{bp_difference:.2f}"
)

st.write(
    f"**Blood Pressure Ratio:** "
    f"{bp_ratio:.2f}"
)


# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🔍 Predict Cardiovascular Disease"
):

    # Create input dictionary

    input_dict = {

        "age_years": age_years,

        "gender": gender,

        "height": height,

        "weight": weight,

        "ap_hi": ap_hi,

        "ap_lo": ap_lo,

        "cholesterol": cholesterol,

        "gluc": gluc,

        "smoke": smoke,

        "alco": alco,

        "active": active,

        "bmi": bmi,

        "bp_difference": bp_difference,

        "bp_ratio": bp_ratio
    }


    # Convert to DataFrame

    input_df = pd.DataFrame(
        [input_dict]
    )


    # Ensure correct feature order

    input_df = input_df[
        feature_columns
    ]


    # Make prediction

    prediction = model.predict(
        input_df
    )[0]


    # Get probability

    probability = model.predict_proba(
        input_df
    )[0][1]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.subheader(
        "Prediction Result"
    )


    if prediction == 1:

        st.error(
            "⚠️ Cardiovascular Disease Predicted"
        )

    else:

        st.success(
            "✅ No Cardiovascular Disease Predicted"
        )


    # Display probability

    st.metric(
        "Predicted Probability",
        f"{probability:.2%}"
    )




