import joblib
import streamlit as st
from risk_engine import evaluate_transaction

st.set_page_config(
    page_title="Razorpay AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Razorpay AI Risk Manager")
st.markdown("**Deterministic ML Fraud Scoring + Rule-Based Risk Engine (No GenAI)**")

@st.cache_resource
def get_categories():
    # Load directly as the Pipeline object
    pipeline = joblib.load("artifacts.joblib")
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    return list(encoder.categories_[0])

try:
    merchant_categories = get_categories()
except Exception:
    merchant_categories = ["Electronics", "Travel", "Grocery", "Retail", "Dining", "Entertainment"]

with st.sidebar:
    st.header("Transaction Inputs")
    amount = st.number_input("Amount (INR ₹)", min_value=1.0, max_value=100000.0, value=250.0, step=25.0)
    transaction_hour = st.slider("Transaction Hour (0–23)", 0, 23, 14)
    merchant_category = st.selectbox("Merchant Category", options=merchant_categories)

    col1, col2 = st.columns(2)
    with col1:
        foreign_transaction = st.radio("Foreign Transaction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        location_mismatch = st.radio("Location Mismatch?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    device_trust_score = st.slider("Device Trust Score (0–100)", 0, 100, 80)
    velocity_last_24h = st.slider("Velocity Last 24h (Tx Count)", 0, 30, 2)
    cardholder_age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35)

payload = {
    "amount": float(amount),
    "transaction_hour": int(transaction_hour),
    "merchant_category": merchant_category,
    "foreign_transaction": int(foreign_transaction),
    "location_mismatch": int(location_mismatch),
    "device_trust_score": int(device_trust_score),
    "velocity_last_24h": int(velocity_last_24h),
    "cardholder_age": int(cardholder_age)
}

st.subheader("Transaction Parameters Under Review")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Amount", f"₹{amount:,.2f}")
m2.metric("Category", merchant_category)
m3.metric("Hour", f"{transaction_hour:02d}:00")
m4.metric("Device Trust", f"{device_trust_score}/100")

st.markdown("---")

if st.button("Evaluate Payment Risk", type="primary", use_container_width=True):
    result = evaluate_transaction(payload)

    r1, r2, r3 = st.columns(3)
    r1.metric("ML Fraud Score", f"{result['fraud_probability'] * 100:.2f}%")
    r2.metric("Pipeline Tier", result["tier"])

    verdict = result["verdict"]
    if verdict == "APPROVE":
        r3.success(f"Verdict: {verdict}")
    elif verdict == "BLOCK":
        r3.error(f"Verdict: {verdict}")
    else:
        r3.warning(f"Verdict: {verdict}")

    st.markdown("### Decision Breakdown & Risk Signals")
    st.write(f"**Recommended Action:** `{result['action']}`")
    st.info(f"**Reasoning / Audit Trail:** {result['reasoning']}")
    st.progress(min(result["fraud_probability"], 1.0))
