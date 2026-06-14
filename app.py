import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ================= UI CONFIG =================
st.set_page_config(
    page_title="MindCheck AI",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>

/* LIGHT GREENISH BACKGROUND (VERY SUBTLE) */
.stApp {
    background: linear-gradient(135deg, #f7fff7, #ecfdf5);
    color: #0f172a;
}

/* TITLE */
h1 {
    color: #14532d;
    text-align: center;
    font-weight: 800;
}

/* GLASSMORPHISM CARD */
.card {
    background: rgba(255, 255, 255, 0.55);
    border-radius: 18px;
    padding: 18px;
    margin: 12px 0;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(34, 197, 94, 0.15);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
    transition: 0.2s;
}

/* SLIDERS */
.stSlider > div > div > div {
    background: #bbf7d0;
}

/* LABELS */
label {
    color: #14532d !important;
    font-weight: 500;
}

/* RADIO TEXT */
.stRadio label {
    color: #14532d !important;
}

/* SUCCESS BOX */
.success-box {
    background: #dcfce7;
    padding: 14px;
    border-radius: 12px;
    color: #14532d;
    border: 1px solid #86efac;
}

/* WARNING BOX */
.warning-box {
    background: #fef9c3;
    padding: 14px;
    border-radius: 12px;
    color: #713f12;
    border: 1px solid #fde047;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("final_student_stress_model.pkl")
feature_columns = joblib.load("final_feature_columns.pkl")

stress_labels = {
    0: "Low Stress",
    1: "Medium Stress",
    2: "High Stress"
}

# ================= TITLE =================
st.title("🧠 MindCheck AI")
st.caption("AI-powered Student Stress Prediction System")
st.write("Fill the questionnaire below to analyze stress level.")

st.divider()

# ================= FORM =================
with st.form("stress_form"):

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🧠 Psychological Factors")
    anxiety_level = st.slider("Anxiety Level", 0, 5, 2)
    self_esteem = st.slider("Self Esteem", 0, 5, 2)
    depression = st.slider("Depression Level", 0, 5, 2)
    mental_health_history = st.radio("Mental Health History", ["No", "Yes"])
    mental_health_history = 1 if mental_health_history == "Yes" else 0
    
    st.markdown('</div><div class="card">', unsafe_allow_html=True)
    st.header("🩺 Physical Factors")
    headache = st.slider("Headache Frequency", 0, 5, 2)
    blood_pressure = st.slider("Blood Pressure Issues", 0, 5, 2)
    sleep_quality = st.slider("Sleep Quality", 0, 5, 2)
    breathing_problem = st.slider("Breathing Problems", 0, 5, 2)

    st.markdown('</div><div class="card">', unsafe_allow_html=True)
    st.header("🌍 Environmental Factors")
    noise_level = st.slider("Noise Level", 0, 5, 2)
    living_conditions = st.slider("Living Conditions", 0, 5, 2)
    safety = st.slider("Safety Level", 0, 5, 2)
    basic_needs = st.slider("Basic Needs", 0, 5, 2)

    st.markdown('</div><div class="card">', unsafe_allow_html=True)
    st.header("🎓 Academic Factors")
    academic_performance = st.slider("Academic Performance", 0, 5, 2)
    study_load = st.slider("Study Load", 0, 5, 2)
    teacher_student_relationship = st.slider("Teacher-Student Relationship", 0, 5, 2)
    future_career_concerns = st.slider("Future Career Concerns", 0, 5, 2)

    st.markdown('</div><div class="card">', unsafe_allow_html=True)
    st.header("🤝 Social Factors")
    social_support = st.slider("Social Support", 0, 5, 2)
    peer_pressure = st.slider("Peer Pressure", 0, 5, 2)
    extracurricular_activities = st.slider("Extracurricular Activities", 0, 5, 2)
    bullying = st.slider("Bullying Experience", 0, 5, 2)
    st.markdown('</div>', unsafe_allow_html=True)

    submit = st.form_submit_button("Predict Stress Level")
    
# ================= FEATURE ENGINEERING =================
def create_features(data):
    df = pd.DataFrame([data])

    df["psychological_health_index"] = df["anxiety_level"] + df["depression"] - df["self_esteem"]
    df["academic_pressure_index"] = df["study_load"] + df["future_career_concerns"] - df["academic_performance"]
    df["mental_health_burden"] = df["anxiety_level"] + df["depression"] + df["mental_health_history"]
    df["lifestyle_balance"] = df["sleep_quality"] + df["social_support"] + df["extracurricular_activities"] - df["study_load"]
    df["social_pressure_index"] = df["peer_pressure"] + df["bullying"] - df["social_support"]
    df["physical_health_risk"] = df["headache"] + df["blood_pressure"] + df["breathing_problem"]
    df["environment_quality_index"] = df["living_conditions"] + df["safety"] + df["basic_needs"] - df["noise_level"]

    return df

# ================= SAVE DATA =================
def save_data(data, prediction):
    data["prediction"] = prediction
    data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([data])
    file = "responses.csv"

    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)

# ================= PREDICTION =================
if submit:

    user_data = {
        "anxiety_level": anxiety_level,
        "self_esteem": self_esteem,
        "mental_health_history": mental_health_history,
        "depression": depression,
        "headache": headache,
        "blood_pressure": blood_pressure,
        "sleep_quality": sleep_quality,
        "breathing_problem": breathing_problem,
        "noise_level": noise_level,
        "living_conditions": living_conditions,
        "safety": safety,
        "basic_needs": basic_needs,
        "academic_performance": academic_performance,
        "study_load": study_load,
        "teacher_student_relationship": teacher_student_relationship,
        "future_career_concerns": future_career_concerns,
        "social_support": social_support,
        "peer_pressure": peer_pressure,
        "extracurricular_activities": extracurricular_activities,
        "bullying": bullying
    }

    df = create_features(user_data)
    df = df.reindex(columns=feature_columns, fill_value=0)

    with st.spinner("Analyzing mental health indicators..."):
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]

    label = stress_labels.get(int(prediction), "Unknown")
    confidence = np.max(probabilities) * 100

    save_data(user_data, label)

    # ================= RISK METRICS =================
    sleep_risk = 5 - sleep_quality
    anxiety_risk = anxiety_level
    academic_risk = study_load + future_career_concerns
    social_risk = (5 - social_support) + peer_pressure + bullying
    overall_risk = np.mean([sleep_risk, anxiety_risk, academic_risk, social_risk])

    def scale(x, m): return min((x / m) * 100, 100)

    risk_values = {
        "Sleep Risk 😴": scale(sleep_risk, 5),
        "Anxiety Risk ": scale(anxiety_risk, 5),
        "Academic Pressure ": scale(academic_risk, 10),
        "Social Stress ": scale(social_risk, 15),
        "⚠️ Overall Stress Level": scale(overall_risk, 5)
    }

    # ================= RESULT =================
    st.subheader("🧠 Wellness Prediction Report")

    if label == "High Stress":
        st.error("🔴 High Stress Detected")
    elif label == "Medium Stress":
        st.warning("🟡 Moderate Stress Detected")
    else:
        st.success("🟢 Low Stress Detected")

    st.markdown("### 🧾 Stress Evaluation Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>📌 Status</h4>
            <h2 style="color:#16a34a;">{label}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>📊 Confidence</h4>
            <h2 style="color:#0ea5e9;">{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h4>🧠 Interpretation</h4>
        <p>
        The model predicts <b>{label}</b> based on psychological, academic, physical and social stress indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ================= INSIGHTS =================
    st.subheader("🧠 Key Stress Indicators")

    insights = []

    if sleep_risk >= 3:
        insights.append("Poor sleep pattern detected")
    if anxiety_risk >= 4:
        insights.append("High anxiety levels detected")
    if academic_risk >= 7:
        insights.append("High academic pressure detected")
    if social_risk >= 8:
        insights.append("Low social support detected")

    for i in insights:
        st.write("•", i)

    if not insights:
        st.success("No major risk indicators detected")

    # ================= RISK GRAPH =================
    st.subheader("🏥 Stress Factor Breakdown")

    fig2, ax2 = plt.subplots()
    bars = ax2.barh(list(risk_values.keys()), list(risk_values.values()))

    for b in bars:
        if b.get_width() > 70:
            b.set_color("red")
        elif b.get_width() > 40:
            b.set_color("orange")
        else:
            b.set_color("green")

    ax2.set_xlim(0, 100)
    st.pyplot(fig2)

    # ================= DISCLAIMER =================
    st.markdown("""
    ---
    ⚠️ **Medical Disclaimer**  
    This tool is for educational purposes only and is NOT a medical diagnosis system.
    """)