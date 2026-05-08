import streamlit as st
import pickle
import numpy as np

# page settings
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# custom css
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #0F172A;
}

.stButton>button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #1D4ED8;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# load model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# title
st.title("🩺 Diabetes Prediction System")

st.markdown(
    "<h4 style='text-align:center;color:gray;'>"
    "Machine Learning Based Health Prediction"
    "</h4>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1505751172876-fa1923c5c528');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("")

# patient details
st.subheader("Patient Details")

Age = st.number_input(
    'Age',
    min_value=1,
    max_value=120
)

Gender = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

Polyuria = st.selectbox(
    'Polyuria',
    ['Yes', 'No']
)

Polydipsia = st.selectbox(
    'Polydipsia',
    ['Yes', 'No']
)

SuddenWeightLoss = st.selectbox(
    'Sudden Weight Loss',
    ['Yes', 'No']
)

Weakness = st.selectbox(
    'Weakness',
    ['Yes', 'No']
)

Polyphagia = st.selectbox(
    'Polyphagia',
    ['Yes', 'No']
)

GenitalThrush = st.selectbox(
    'Genital Thrush',
    ['Yes', 'No']
)

VisualBlurring = st.selectbox(
    'Visual Blurring',
    ['Yes', 'No']
)

Itching = st.selectbox(
    'Itching',
    ['Yes', 'No']
)

Irritability = st.selectbox(
    'Irritability',
    ['Yes', 'No']
)

DelayedHealing = st.selectbox(
    'Delayed Healing',
    ['Yes', 'No']
)

PartialParesis = st.selectbox(
    'Partial Paresis',
    ['Yes', 'No']
)

MuscleStiffness = st.selectbox(
    'Muscle Stiffness',
    ['Yes', 'No']
)

Alopecia = st.selectbox(
    'Alopecia',
    ['Yes', 'No']
)

Obesity = st.selectbox(
    'Obesity',
    ['Yes', 'No']
)

# encoding
Gender = 0 if Gender == 'Male' else 1

def yn(x):
    return 0 if x == 'Yes' else 1

# prediction
if st.button('Predict Diabetes'):

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

    # model prediction
    prediction = loaded_model.predict(input_data)

    # probability prediction
    probability = loaded_model.predict_proba(input_data)

    diabetic_probability = probability[0][1] * 100

    # risk level
    if diabetic_probability < 30:
        risk = "Low Risk"
        risk_color = "#BBF7D0"
        text_color = "#166534"

    elif diabetic_probability < 70:
        risk = "Moderate Risk"
        risk_color = "#FDE68A"
        text_color = "#92400E"

    else:
        risk = "High Risk"
        risk_color = "#FECACA"
        text_color = "#991B1B"

    st.write("")

    # positive result
    if prediction[0] == 'Positive':

        st.markdown(
            f"""
            <div style='
            background-color:{risk_color};
            padding:25px;
            border-radius:15px;
            text-align:center;
            '>

            <h2 style='color:{text_color};'>
            ⚠️ Diabetic Positive
            </h2>

            <h3 style='color:{text_color};'>
            Risk Level: {risk}
            </h3>

            <h3 style='color:{text_color};'>
            Probability: {diabetic_probability:.2f}%
            </h3>

            </div>
            """,
            unsafe_allow_html=True
        )

    # negative result
    else:

        st.markdown(
            f"""
            <div style='
            background-color:#BBF7D0;
            padding:25px;
            border-radius:15px;
            text-align:center;
            '>

            <h2 style='color:#166534;'>
            ✅ Not Diabetic
            </h2>

            <h3 style='color:#166534;'>
            Risk Level: {risk}
            </h3>

            <h3 style='color:#166534;'>
            Probability: {diabetic_probability:.2f}%
            </h3>

            </div>
            """,
            unsafe_allow_html=True
        )
