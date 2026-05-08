import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        to right,
        #eef4ff,
        #f8fbff
    );
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #0f172a;
    margin-bottom: 0px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 25px;
    color: #64748b;
    margin-top: 0px;
    margin-bottom: 40px;
}

/* Card */
.card {
    background-color: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

/* Section Heading */
.section-title {
    font-size: 35px;
    font-weight: bold;
    color: #0f172a;
    margin-bottom: 20px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 12px;
    border: none;
    font-size: 22px;
    font-weight: bold;
    color: white;
    background: linear-gradient(
        to right,
        #2563eb,
        #7c3aed
    );
}

.stButton > button:hover {
    color: white;
}

/* Result Box */
.result-box {
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 30px;
}

/* Disclaimer */
.disclaimer {
    background-color: #eff6ff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
    color: #1e3a8a;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

loaded_model = pickle.load(
    open('diabetes_model.sav', 'rb')
)

# ---------------- TITLE ----------------

st.markdown(
    """
    <div class='main-title'>
    🩺 Diabetes Prediction System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
    Machine Learning Based Health Prediction
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- CARD START ----------------

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='section-title'>
    👤 Patient Details
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- INPUTS ----------------

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input(
        'Age',
        min_value=1,
        max_value=120
    )

with col2:
    Gender = st.selectbox(
        'Gender',
        ['Male', 'Female']
    )

# Row 1
c1, c2, c3 = st.columns(3)

with c1:
    Polyuria = st.selectbox(
        'Polyuria',
        ['Yes', 'No']
    )

with c2:
    Polydipsia = st.selectbox(
        'Polydipsia',
        ['Yes', 'No']
    )

with c3:
    SuddenWeightLoss = st.selectbox(
        'Sudden Weight Loss',
        ['Yes', 'No']
    )

# Row 2
c1, c2, c3 = st.columns(3)

with c1:
    Weakness = st.selectbox(
        'Weakness',
        ['Yes', 'No']
    )

with c2:
    Polyphagia = st.selectbox(
        'Polyphagia',
        ['Yes', 'No']
    )

with c3:
    GenitalThrush = st.selectbox(
        'Genital Thrush',
        ['Yes', 'No']
    )

# Row 3
c1, c2, c3 = st.columns(3)

with c1:
    VisualBlurring = st.selectbox(
        'Visual Blurring',
        ['Yes', 'No']
    )

with c2:
    Itching = st.selectbox(
        'Itching',
        ['Yes', 'No']
    )

with c3:
    Irritability = st.selectbox(
        'Irritability',
        ['Yes', 'No']
    )

# Row 4
c1, c2, c3 = st.columns(3)

with c1:
    DelayedHealing = st.selectbox(
        'Delayed Healing',
        ['Yes', 'No']
    )

with c2:
    PartialParesis = st.selectbox(
        'Partial Paresis',
        ['Yes', 'No']
    )

with c3:
    MuscleStiffness = st.selectbox(
        'Muscle Stiffness',
        ['Yes', 'No']
    )

# Row 5
c1, c2 = st.columns(2)

with c1:
    Alopecia = st.selectbox(
        'Alopecia',
        ['Yes', 'No']
    )

with c2:
    Obesity = st.selectbox(
        'Obesity',
        ['Yes', 'No']
    )

# ---------------- ENCODING ----------------

Gender = 1 if Gender == 'Male' else 0

def yn(value):
    return 1 if value == 'Yes' else 0

# ---------------- PREDICTION ----------------

st.write("")

if st.button('⚡ Predict Diabetes'):

    input_data = np.asarray([[
        Age,
        Gender,
        yn(Polyuria),
        yn(Polydipsia),
        yn(SuddenWeightLoss),
        yn(Weakness),
        yn(Polyphagia),
        yn(GenitalThrush),
        yn(VisualBlurring),
        yn(Itching),
        yn(Irritability),
        yn(DelayedHealing),
        yn(PartialParesis),
        yn(MuscleStiffness),
        yn(Alopecia),
        yn(Obesity)
    ]])

    # prediction
    prediction = loaded_model.predict(input_data)

    # probability
    probability = loaded_model.predict_proba(
        input_data
    )

    diabetic_probability = (
        probability[0][1] * 100
    )

    # risk level
    if diabetic_probability < 30:
        risk = "Low Risk"

    elif diabetic_probability < 70:
        risk = "Moderate Risk"

    else:
        risk = "High Risk"

    # ---------------- POSITIVE ----------------

    if prediction[0] == 1:

        st.markdown(
            f"""
            <div class='result-box'
            style='
            background-color:#fee2e2;
            color:#991b1b;
            '>

            ⚠️ Diabetic Positive<br><br>

            Risk Level:
            {risk}<br><br>

            Probability:
            {diabetic_probability:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- NEGATIVE ----------------

    else:

        st.markdown(
            f"""
            <div class='result-box'
            style='
            background-color:#dcfce7;
            color:#166534;
            '>

            ✅ Not Diabetic<br><br>

            Risk Level:
            {risk}<br><br>

            Probability:
            {diabetic_probability:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- DISCLAIMER ----------------

st.markdown(
    """
    <div class='disclaimer'>
    ⚠️ Disclaimer:
    This prediction is based on machine learning
    and should not replace professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- CARD END ----------------

st.markdown(
    "</div>",
    unsafe_allow_html=True
)
