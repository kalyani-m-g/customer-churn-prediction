import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------- LOAD FILES ---------------- #

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color:#0F172A;
}

.stButton > button {
    width:100%;
    height:60px;
    border-radius:12px;
    background:linear-gradient(90deg,#059669,#14B8A6);
    color:white;
    font-size:20px;
    font-weight:bold;
    border:none;
}

.stButton > button:hover{
    transform:scale(1.02);
}

[data-testid="stMetric"]{
    background:#111827;
    border:1px solid #1F2937;
    border-radius:12px;
    padding:15px;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🏦 Banking Analytics")

    st.info("""
### Customer Churn Prediction

Predict whether a customer is likely to leave the bank based on customer demographics, account information and activity patterns.
""")

    st.markdown("""
### 📊 Project Overview

- Machine Learning Based Prediction
- Customer Retention Analytics
- Probability-Based Risk Assessment
- Banking Decision Support Tool

### 🔍 Features Used

- Customer Demographics
- Credit Information
- Banking Activity
- Account Balance
- Customer Engagement
""")

    st.warning("""
⚠️ Educational Project

This application is intended for educational and portfolio purposes only.
""")

    st.markdown("---")

    st.write("👩‍💻 Developed by")
    st.write("**Kalyani M G**")

# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;color:#14B8A6;'>
🏦 Customer Churn Prediction Dashboard
</h1>

<p style='text-align:center;font-size:18px;color:#94A3B8;'>
Customer Retention & Risk Analytics Platform
</p>
""", unsafe_allow_html=True)

st.markdown("---")

k1, k2, k3 = st.columns(3)

with k1:
    st.metric("Features Used", "11")

with k2:
    st.metric("Model Type", "Gradient Boosting")

with k3:
    st.metric("Industry", "Banking")

st.warning("""
⚠️ Disclaimer

Predictions are generated using a machine learning model and should be used only as a decision-support tool.
""")

# ---------------- INPUTS ---------------- #

st.markdown("""
### 👤 Customer Information

Enter customer demographic and banking details to evaluate churn risk.
""")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    geography = st.selectbox(
        "Country",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["No", "Yes"]
    )

    active_member = st.selectbox(
        "Active Member",
        ["No", "Yes"]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

# ---------------- ENCODING ---------------- #

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

has_card_val = 1 if has_card == "Yes" else 0
active_member_val = 1 if active_member == "Yes" else 0

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict Churn Risk"):

    customer = pd.DataFrame([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card_val,
        active_member_val,
        salary,
        geo_germany,
        geo_spain,
        gender_male
    ]], columns=feature_names)

    customer_scaled = scaler.transform(customer)

    prediction = model.predict(customer_scaled)

    probability = model.predict_proba(customer_scaled)[0][1]

    st.markdown("---")

    st.subheader("📊 Prediction Results")

    st.markdown(f"""
    <div style="
    background:linear-gradient(90deg,#064E3B,#0F766E);
    padding:25px;
    border-radius:15px;
    color:white;
    margin-bottom:20px;
    ">
        <h4 style="margin:0;">
        Churn Probability
        </h4>

        <h1 style="margin-top:10px;">
        {probability*100:.2f}%
        </h1>

    </div>
    """, unsafe_allow_html=True)

    st.progress(float(probability))

    if probability < 0.30:

        st.success(
            "🟢 Low Risk: Customer is likely to stay."
        )

    elif probability < 0.70:

        st.warning(
            "🟡 Moderate Risk: Customer retention actions may be beneficial."
        )

    else:

        st.error(
            "🔴 High Risk: Customer is likely to churn."
        )

    if prediction[0] == 1:
            

        st.error(
            "Prediction: Customer Will Churn"
        )

    else:

        st.success(
            "Prediction: Customer Will Stay"
        )

    st.markdown("### 💡 Retention Recommendation")

    if probability < 0.30:

        st.success(
            "Customer appears loyal. Continue maintaining service quality and engagement."
        )

    elif probability < 0.70:

        st.warning(
            "Customer may be at risk. Consider personalized offers, rewards or proactive communication."
        )

    else:

        st.error(
            "Immediate retention action recommended. Customer shows strong churn indicators."
        )

# ---------------- PROJECT INFO ---------------- #

st.markdown("---")

with st.expander("📖 View Project Information"):

    st.write("""
### Customer Churn Prediction System

This application uses machine learning to predict whether a customer is likely to leave a banking institution.

The model analyzes customer demographics, banking behavior, account characteristics and engagement indicators to estimate churn probability.

### Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Gradient Boosting Classifier

### Key Features

- Customer Risk Assessment
- Churn Probability Estimation
- Interactive Dashboard
- Real-Time Prediction

### Dataset

Bank Customer Churn Prediction Dataset

### Developer

Kalyani M G
""")

st.markdown("---")

st.caption(
    "© 2026 Customer Churn Prediction System | Developed by Kalyani M G"
)
