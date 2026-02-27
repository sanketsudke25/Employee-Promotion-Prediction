import streamlit as st
import pickle
import pandas as pd

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
    background: linear-gradient(135deg, #141E30, #243B55);
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    text-align: center;
    color: white;
    font-size: 36px;
    font-weight: 700;
}

.block-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

label {
    color: #ffffff !important;
    font-weight: 600;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #dfe6e9;
    font-size: 14px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("promotion_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# -------------------- TITLE --------------------
st.title("📈 Employee Promotion Prediction")
st.markdown("<h3 style='color:white;text-align:center;'>Enter Employee Details</h3>", unsafe_allow_html=True)

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department", [
        "Sales & Marketing", "Operations", "Technology",
        "Analytics", "R&D", "Finance", "HR",
        "Legal", "Procurement"
    ])

    region = st.selectbox("Region", [
        "region_1", "region_2", "region_3", "region_4",
        "region_5", "region_6", "region_7", "region_8"
    ])

    education = st.selectbox("Education", [
        "Below Secondary", "Bachelor's", "Master's & above"
    ])

    gender = st.selectbox("Gender", ["m", "f"])

    recruitment_channel = st.selectbox("Recruitment Channel", [
        "sourcing", "referred", "other"
    ])

    no_of_trainings = st.number_input("Number of Trainings", min_value=0, value=1)

with col2:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)

    previous_year_rating = st.slider("Previous Year Rating", 1, 5, 3)

    length_of_service = st.number_input("Length of Service", min_value=0, value=5)

    awards_won = st.selectbox("Awards Won?", [0, 1])  # 🔥 ADDED

    kpi = st.selectbox("KPIs Met >80%", [0, 1])

    avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=60)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Promotion Status"):

    input_data = pd.DataFrame([{
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "awards_won": awards_won,   # 🔥 FIXED
        "KPIs_met >80%": kpi,
        "avg_training_score": avg_training_score
    }])

    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)

    if prediction[0] == 1:
        st.success("🎉 Employee WILL be Promoted!")
        st.info(f"Confidence: {probability[0][1]*100:.2f}%")
    else:
        st.error("❌ Employee Will NOT be Promoted")
        st.info(f"Confidence: {probability[0][0]*100:.2f}%")

# -------------------- FOOTER --------------------
st.markdown("<div class='footer'>Model: Random Forest | Metric Focus: Recall | Developed by Sanket Sudke 🚀</div>", unsafe_allow_html=True)
