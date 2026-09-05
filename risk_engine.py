import joblib
import pandas as pd

ARTIFACTS_PATH = r"C:\credit_card_fraud\artifacts.joblib"
pipeline = joblib.load(ARTIFACTS_PATH)

def generate_rule_based_audit(tx_dict: dict) -> str:
    reasons = []

    if tx_dict.get("location_mismatch") == 1:
        reasons.append("IP / Billing location mismatch detected")
    if tx_dict.get("velocity_last_24h", 0) >= 5:
        reasons.append(f"Elevated velocity ({tx_dict['velocity_last_24h']} transactions in 24h)")
    if tx_dict.get("device_trust_score", 100) < 50:
        reasons.append(f"Untrusted device signature (Trust score: {tx_dict['device_trust_score']}/100)")
    if tx_dict.get("foreign_transaction") == 1:
        reasons.append("Cross-border / international payment corridor")
    if tx_dict.get("amount", 0) > 10000:
        reasons.append(f"High ticket size (₹{tx_dict['amount']:,.2f})")

    if not reasons:
        return "Multiple marginal baseline anomalies triggered an edge-case review."
    return "; ".join(reasons) + "."


def evaluate_transaction(tx_dict: dict) -> dict:
    # Pass input dictionary straight into DataFrame for pipeline prediction
    df_row = pd.DataFrame([tx_dict])
    ml_prob = float(pipeline.predict_proba(df_row)[0][1])

    if ml_prob < 0.30:
        return {
            "tier": "Tier-1 (Deterministic ML)",
            "fraud_probability": round(ml_prob, 4),
            "verdict": "APPROVE",
            "action": "Instant Settlement",
            "reasoning": "Transaction features align with normal consumer baseline."
        }

    if ml_prob > 0.75:
        return {
            "tier": "Tier-1 (Deterministic ML)",
            "fraud_probability": round(ml_prob, 4),
            "verdict": "BLOCK",
            "action": "Immediate Block & Log Incident",
            "reasoning": "High-confidence anomaly detected across velocity, device trust, or location parameters."
        }

    audit_explanation = generate_rule_based_audit(tx_dict)
    return {
        "tier": "Tier-2 (Rule-Based Risk Engine)",
        "fraud_probability": round(ml_prob, 4),
        "verdict": "STEP_UP_2FA",
        "action": "Dispatch SMS / Biometric 2FA Challenge",
        "reasoning": audit_explanation
    }