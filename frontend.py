import json
import httpx
import pandas as pd
import streamlit as st

# ---------- Page config and sidebar ----------
st.set_page_config(page_title="Loan Approval", page_icon="✅", layout="wide")

st.sidebar.header("Settings")
API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/api/loan/v1/predict")
timeout_s = st.sidebar.slider("Request timeout (s)", 2, 30, 10)
show_prob_as = st.sidebar.selectbox("Show probability as", ["Percent", "Decimal"])

# ---------- Helpers ----------
def predict_loan_approval(payload: dict) -> dict:
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(API_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

def as_percent(p: float) -> str:
    try:
        return f"{100.0 * float(p):.2f} %"
    except Exception:
        return "N/A"

# ---------- Title ----------
st.title("Loan Approval Prediction")

# ---------- Tabs ----------
tab_form, tab_preview, tab_about = st.tabs(["Form", "Preview / Custom JSON", "About"])

with tab_form:
    st.subheader("Applicant Information")

    # Arrange inputs in responsive columns
    c1, c2, c3 = st.columns([1, 1, 1])
    c4, c5, c6 = st.columns([1, 1, 1])
    c7, c8, c9 = st.columns([1, 1, 1])
    c10, c11 = st.columns([1, 1])

    with c1:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, step=1)
    with c2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c3:
        self_employed = st.radio("Self Employed", ["Yes", "No"], horizontal=True)
    with c4:
        income_annum = st.number_input("Annual Income", min_value=200_000, max_value=9_900_000, step=100_000)
    with c5:
        loan_amount = st.number_input("Loan Amount", min_value=300_000, max_value=39_500_000, step=100_000)
    with c6:
        loan_term = st.number_input("Loan Term (years)", min_value=2, max_value=20, step=1)
    with c7:
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    with c8:
        residential_assets_value = st.number_input("Residential Assets Value", min_value=-100_000, max_value=29_100_000, step=100_000)
    with c9:
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, max_value=19_400_000, step=100_000)
    with c10:
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=300_000, max_value=39_200_000, step=100_000)
    with c11:
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, max_value=14_700_000, step=100_000)

    # Build API payload (use aliases expected by backend)
    payload = {
        "no_of_dependents": int(no_of_dependents),
        "education": education,
        "selfEmployed": self_employed,
        "incomeAnnum": float(income_annum),
        "loanAmount": float(loan_amount),
        "loanTerm": float(loan_term),
        "cibilScore": float(cibil_score),
        "residentialAssetsValue": float(residential_assets_value),
        "commercialAssetsValue": float(commercial_assets_value),
        "luxuryAssetsValue": float(luxury_assets_value),
        "bankAssetValue": float(bank_asset_value),
    }

    # Predict
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Contacting API..."):
                try:
                    result = predict_loan_approval(payload)
                    approved = result.get("approved")
                    probability = result.get("probability", None)

                    st.success("Decision received")
                    if probability is not None:
                        if show_prob_as == "Percent":
                            st.metric("Approval probability", as_percent(probability))
                        else:
                            st.metric("Approval probability", f"{float(probability):.4f}")

                    if str(approved).strip().lower() in {"1", "approved", "true", "yes"}:
                        st.success("Loan Approved!")
                        st.balloons()
                    else:
                        st.error("Loan Not Approved.")
                except httpx.HTTPError as e:
                    st.error(f"API error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

with tab_preview:
    st.subheader("Custom / Preview JSON")
    st.caption("Paste or edit your JSON payload")

    user_json = st.text_area("Enter JSON here", value=json.dumps(payload, indent=2), height=300)

    try:
        custom_payload = json.loads(user_json)
        st.success("Valid JSON ✅")
        st.json(custom_payload)

        if st.button("Send Custom JSON", type="secondary"):
            with st.spinner("Contacting API..."):
                try:
                    result = predict_loan_approval(custom_payload)
                    st.write("### API Response:")
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON ❌: {e}")

with tab_about:
    st.subheader("About")
    st.write("This app calls a FastAPI endpoint with a trained pipeline to predict loan approvals.")
    st.write("Use the Settings sidebar to change the API URL and timeout if the backend runs elsewhere.")
