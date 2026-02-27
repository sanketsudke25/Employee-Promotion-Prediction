import streamlit as st
import pickle
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="📈",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 34px;
    }

    h3 {
        color: #dfe6e9;
    }

    label {
        color: white !important;
        font-weight: 600;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
    }

    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("promotion_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------- TITLE --------------------
st.title("📈 Employee Promotion Prediction")
st.write("### Enter Employee Details Below")

# -------------------- INPUT LAYOUT --------------------
col1, col2 = st.columns(2)

with col1:
    department = st.number_input("Department (Encoded)", min_value=0)
    region = st.number_input("Region (Encoded)", min_value=0)
    education = st.number_input("Education (Encoded)", min_value=0)
    gender = st.number_input("Gender (Encoded)", min_value=0)
    recruitment_channel = st.number_input("Recruitment Channel (Encoded)", min_value=0)
    no_of_trainings = st.number_input("Number of Trainings", min_value=0)

with col2:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    previous_year_rating = st.slider("Previous Year Rating", 1, 5, 3)
    length_of_service = st.number_input("Length of Service", min_value=0, value=5)
    kpi = st.selectbox("KPIs Met >80%", [0, 1])
    avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=60)

# -------------------- PREDICTION --------------------
if st.button("Predict Promotion Status"):

    input_data = np.array([[department,
                            region,
                            education,
                            gender,
                            recruitment_channel,
                            no_of_trainings,
                            age,
                            previous_year_rating,
                            length_of_service,
                            kpi,
                            avg_training_score]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success(f"🎉 Employee WILL be Promoted!")
        st.info(f"Confidence: {probability[0][1]*100:.2f}%")
    else:
        st.error("❌ Employee Will NOT be Promoted")
        st.info(f"Confidence: {probability[0][0]*100:.2f}%")

# -------------------- MODEL INFO --------------------
st.markdown("---")
st.write("🔍 **Model Used:** Random Forest Classifier")
st.write("📊 **Model Accuracy:** 94% (Example - replace with your actual accuracy)")