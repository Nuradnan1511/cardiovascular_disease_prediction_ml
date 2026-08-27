

import streamlit as st
import pandas as pd
import joblib



# PAGE CONFIGURATION

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CSS

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Section headers */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Information cards */
    .info-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f7f9fc;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .info-label {
        font-size: 14px;
        color: #666;
    }

    .info-value {
        font-size: 28px;
        font-weight: 700;
    }

    /* Prediction box */
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #f7f9fc;
        border: 1px solid #e5e7eb;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# LOAD MODEL

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/cardiovascular_disease_model.pkl"
    )

    feature_columns = joblib.load(
        "models/feature_columns.pkl"
    )

    return model, feature_columns


model, feature_columns = load_model()






# HEADER

st.markdown(
    '<div class="main-title">❤️ Cardiovascular Disease Prediction</div>',
    unsafe_allow_html=True
)

##st.markdown(
    ##'<div class="subtitle">'
    ##'Machine Learning-Based Cardiovascular Risk Prediction'
    ##'</div>',
   ## unsafe_allow_html=True
##)


st.info(
    "Enter the patient's information below to "
    "generate a machine learning prediction."
)



# PATIENT INFORMATION

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    age_years = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    gender_option = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    if gender_option == "Male":
        gender = 1
    else:
        gender = 2


with col2:

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=1.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=70.0,
        step=1.0
    )



# BLOOD PRESSURE

st.markdown(
    '<div class="section-title">🩺 Blood Pressure</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    ap_hi = st.number_input(
        "Systolic Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1,
        help="Upper blood pressure value."
    )


with col2:

    ap_lo = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30,
        max_value=150,
        value=80,
        step=1,
        help="Lower blood pressure value."
    )



# HEALTH INFORMATION

st.markdown(
    '<div class="section-title">🏥 Health & Lifestyle</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

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


with col2:

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


with col3:

    smoke_option = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    smoke = 1 if smoke_option == "Yes" else 0


col1, col2 = st.columns(2)


with col1:

    alcohol_option = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    alco = 1 if alcohol_option == "Yes" else 0


with col2:

    activity_option = st.selectbox(
        "Physical Activity",
        ["No", "Yes"]
    )

    active = 1 if activity_option == "Yes" else 0



# FEATURE ENGINEERING

bmi = weight / ((height / 100) ** 2)

bp_difference = ap_hi - ap_lo

bp_ratio = ap_hi / ap_lo



# CALCULATED FEATURES

st.markdown(
    '<div class="section-title">📊 Calculated Health Metrics</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "BMI",
        f"{bmi:.2f}"
    )


with col2:

    st.metric(
        "BP Difference",
        f"{bp_difference:.0f}"
    )


with col3:

    st.metric(
        "BP Ratio",
        f"{bp_ratio:.2f}"
    )



# PREDICTION BUTTON

st.divider()

predict_button = st.button(
    "🔍 Predict Cardiovascular Disease",
    use_container_width=True,
    type="primary"
)



# PREDICTION

if predict_button:

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


    # Probability

    probability = model.predict_proba(
        input_df
    )[0][1]


   
    # DISPLAY RESULT
    # ======================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.error(
            "⚠️ Cardiovascular Disease Predicted"
        )

    else:

        st.success(
            "✅ No Cardiovascular Disease Predicted"
        )


    # Probability

    st.metric(
        "Predicted Probability",
        f"{probability:.2%}"
    )


    # Progress bar

    st.progress(
        float(probability)
    )


    # Interpretation

    if probability >= 0.70:

        st.warning(
            "The model estimates a relatively high "
            "probability of cardiovascular disease."
        )

    elif probability >= 0.50:

        st.info(
            "The model estimates a moderate probability "
            "of cardiovascular disease."
        )

    else:

        st.success(
            "The model estimates a relatively low "
            "probability of cardiovascular disease."
        )





# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
    developed by Nur Alam
        
    </div>
    """,
    unsafe_allow_html=True
)