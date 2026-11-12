import pickle
import pandas as pd
import streamlit as st

# Load the trained Random Forest model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below to get a model prediction.")

st.info(
    "This is an educational machine-learning demo. "
    "It is not an actual bank decision or financial advice."
)

with st.form("loan_form"):
    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )

    with col2:
        applicant_income = st.number_input(
            "Applicant Income", min_value=0, value=5000, step=500
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0.0, value=0.0, step=500.0
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands)", min_value=0.0, value=120.0, step=10.0
        )
        loan_term = st.number_input(
            "Loan Amount Term (months)", min_value=1.0, value=360.0, step=12.0
        )
        credit_history = st.selectbox(
            "Credit History", [1.0, 0.0],
            format_func=lambda x: "Good (1)" if x == 1.0 else "Not good (0)"
        )

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    # Convert the selected text values into the same numeric labels used in training.
    input_data = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
        "Education": 0 if education == "Graduate" else 1,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    }])

    # Keep the feature order identical to the training data.
    input_data = input_data[model.feature_names_in_]

    prediction = int(model.predict(input_data)[0])

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The model predicts: Loan Approved")
    else:
        st.error("The model predicts: Loan Not Approved")

    probabilities = model.predict_proba(input_data)[0]
    st.write(f"Model estimated approval score: **{probabilities[1] * 100:.1f}%**")
    st.caption(
        "This score is the model's output, not a real-world probability "
        "or a guarantee of approval."
    )
