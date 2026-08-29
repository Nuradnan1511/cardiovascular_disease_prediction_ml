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
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Calculated metric cards */
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #f7f9fc;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-label {
        font-size: 14px;
        color: #666;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
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


# ==========================================
# LOAD MODEL
# ==========================================

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


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '❤️ Cardiovascular Disease Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Cardiovascular Risk Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the patient's health and lifestyle information "
    "to generate a machine learning prediction."
)


# ==========================================
# PATIENT INFORMATION
# ==========================================

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


# ==========================================
# BLOOD PRESSURE
# ==========================================

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


# ==========================================
# HEALTH & LIFESTYLE
# ==========================================

st.markdown(
    '<div class="section-title">🏥 Health & Lifestyle</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# Cholesterol
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


# Glucose
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


# Smoking
with col3:

    smoke_option = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    smoke = 1 if smoke_option == "Yes" else 0


col1, col2 = st.columns(2)


# Alcohol
with col1:

    alcohol_option = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    alco = 1 if alcohol_option == "Yes" else 0


# Physical Activity
with col2:

    activity_option = st.selectbox(
        "Physical Activity",
        ["No", "Yes"]
    )

    active = 1 if activity_option == "Yes" else 0


# ==========================================
# CALCULATED FEATURES
# ==========================================

bmi = weight / ((height / 100) ** 2)

bp_difference = ap_hi - ap_lo

bp_ratio = ap_hi / ap_lo


# ==========================================
# CALCULATED HEALTH METRICS
# ==========================================

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


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def make_prediction():

    # ======================================
    # INPUT DATA
    # ======================================

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


    # ======================================
    # DATAFRAME
    # ======================================

    input_df = pd.DataFrame(
        [input_dict]
    )


    # ======================================
    # FEATURE ORDER
    # ======================================

    input_df = input_df[
        feature_columns
    ]


    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        input_df
    )[0]


    probability = model.predict_proba(
        input_df
    )[0][1]


    return prediction, probability


# ==========================================
# POPUP RESULT
# ==========================================

@st.dialog("🎯 Prediction Result")
def show_prediction_result(
    prediction,
    probability
):

    # --------------------------------------
    # Result
    # --------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Cardiovascular Disease Predicted"
        )

    else:

        st.success(
            "✅ No Cardiovascular Disease Predicted"
        )


    # --------------------------------------
    # Probability
    # --------------------------------------

    st.subheader(
        "Predicted Probability"
    )

    st.metric(
        "Probability of Cardiovascular Disease",
        f"{probability:.2%}"
    )


    # --------------------------------------
    # Progress Bar
    # --------------------------------------

    st.progress(
        float(probability)
    )


    # --------------------------------------
    # Interpretation
    # --------------------------------------

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


    # --------------------------------------
    # Patient Summary
    # --------------------------------------

    st.divider()

    st.subheader(
        "Patient Summary"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Age:** {age_years} years"
        )

        st.write(
            f"**Gender:** {gender_option}"
        )

        st.write(
            f"**BMI:** {bmi:.2f}"
        )

    with col2:

        st.write(
            f"**Systolic BP:** {ap_hi}"
        )

        st.write(
            f"**Diastolic BP:** {ap_lo}"
        )

        st.write(
            f"**Cholesterol:** {cholesterol_option}"
        )




# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

st.markdown(
    "<div style='text-align:center;'>"
    "<h3>Ready to generate the prediction?</h3>"
    "</div>",
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(
    [1, 2, 1]
)


with col2:

    predict_button = st.button(
        "🔍 Predict Cardiovascular Disease",
        use_container_width=True,
        type="primary"
    )


# ==========================================
# BUTTON ACTION
# ==========================================

if predict_button:

    prediction, probability = make_prediction()

    show_prediction_result(
        prediction,
        probability
    )




# ==========================================
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