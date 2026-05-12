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

Polyuria = st.selectbox('Polyuria', ['Yes', 'No'])
Polydipsia = st.selectbox('Polydipsia', ['Yes', 'No'])
SuddenWeightLoss = st.selectbox('Sudden Weight Loss', ['Yes', 'No'])
Weakness = st.selectbox('Weakness', ['Yes', 'No'])
Polyphagia = st.selectbox('Polyphagia', ['Yes', 'No'])
GenitalThrush = st.selectbox('Genital Thrush', ['Yes', 'No'])
VisualBlurring = st.selectbox('Visual Blurring', ['Yes', 'No'])
Itching = st.selectbox('Itching', ['Yes', 'No'])
Irritability = st.selectbox('Irritability', ['Yes', 'No'])
DelayedHealing = st.selectbox('Delayed Healing', ['Yes', 'No'])
PartialParesis = st.selectbox('Partial Paresis', ['Yes', 'No'])
MuscleStiffness = st.selectbox('Muscle Stiffness', ['Yes', 'No'])
Alopecia = st.selectbox('Alopecia', ['Yes', 'No'])
Obesity = st.selectbox('Obesity', ['Yes', 'No'])

# CORRECT encoding
Gender = 1 if Gender == 'Male' else 0

def yn(x):
    return 1 if x == 'Yes' else 0

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

    prediction = loaded_model.predict(input_data)

st.write("")

if prediction[0] == 1 or prediction[0] == 'Positive':

    st.markdown(
        """
        <div class='result-box'
        style='background-color:#FECACA;color:#991B1B;'>
        ⚠️ Diabetic Positive
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class='result-box'
        style='background-color:#BBF7D0;color:#166534;'>
        ✅ Not Diabetic
        </div>
        """,
        unsafe_allow_html=True
    )

    else:

        st.markdown(
            """
            <div class='result-box'
            style='background-color:#BBF7D0;color:#166534;'>
            ✅ Not Diabetic
            </div>
            """,
            unsafe_allow_html=True
        )
