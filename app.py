import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("voting_model_personal_loan.pkl", "rb"))

# Streamlit app title
st.title("📊 Personal Loan Prediction App")

st.write("Fill in the customer details to predict if they will subscribe to a term deposit.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=40)
balance = st.number_input("Balance", value=1000)
campaign = st.number_input("Campaign (number of contacts)", min_value=1, value=1)
pdays = st.number_input("Pdays (days since last contact, -1 if never contacted)", value=-1)
previous = st.number_input("Previous (number of contacts before this campaign)", min_value=0, value=0)

# Job selection (one-hot encoded in model)
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
               'retired', 'self-employed', 'services', 'student', 'technician',
               'unemployed', 'unknown']
job = st.selectbox("Job", job_options)

# Marital status
marital_options = ['divorced', 'married', 'single']
marital = st.selectbox("Marital Status", marital_options)

# Education
education_options = ['primary', 'secondary', 'tertiary', 'unknown']
education = st.selectbox("Education", education_options)

# Default credit
default = st.radio("Default Credit?", ['yes', 'no'])

# Housing loan
housing = st.radio("Housing Loan?", ['yes', 'no'])

# Personal loan
loan = st.radio("Personal Loan?", ['yes', 'no'])

# Contact communication type
contact_options = ['cellular', 'telephone', 'unknown']
contact = st.selectbox("Contact Communication Type", contact_options)

# Poutcome
poutcome_options = ['failure', 'other', 'success', 'unknown']
poutcome = st.selectbox("Outcome of Previous Campaign", poutcome_options)


# ---------------------------
# Convert inputs into dataframe (with same columns as training)
# ---------------------------
def preprocess_inputs():
    # Initialize all columns with 0
    input_dict = {
        'age': age,
        'balance': balance,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
    }

    # Add one-hot encoded categorical features
    for col in [
        'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
        'job_management', 'job_retired', 'job_self-employed', 'job_services',
        'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
        'marital_divorced', 'marital_married', 'marital_single',
        'education_primary', 'education_secondary', 'education_tertiary',
        'education_unknown', 'default_no', 'default_yes',
        'housing_no', 'housing_yes', 'loan_no', 'loan_yes',
        'contact_cellular', 'contact_telephone', 'contact_unknown',
        'poutcome_failure', 'poutcome_other', 'poutcome_success', 'poutcome_unknown'
    ]:
        input_dict[col] = 0

    # One-hot encoding for job
    input_dict[f'job_{job}'] = 1

    # One-hot encoding for marital
    input_dict[f'marital_{marital}'] = 1

    # One-hot encoding for education
    input_dict[f'education_{education}'] = 1

    # Default
    input_dict[f'default_{default}'] = 1

    # Housing
    input_dict[f'housing_{housing}'] = 1

    # Loan
    input_dict[f'loan_{loan}'] = 1

    # Contact
    input_dict[f'contact_{contact}'] = 1

    # Poutcome
    input_dict[f'poutcome_{poutcome}'] = 1

    return pd.DataFrame([input_dict])


# --- Prediction block (replace your current one) ---
threshold = st.slider(
    "Decision threshold (P(subscribe) ≥ threshold → Predict: subscribe)",
    min_value=0.05, max_value=0.95, value=0.50, step=0.01
)

if st.button("🔮 Predict"):
    X = preprocess_inputs()
    proba = model.predict_proba(X)[0]

    # Find the positive class index robustly
    classes = list(model.classes_)
    if 'yes' in classes:
        pos_idx = classes.index('yes')
    elif 1 in classes:
        pos_idx = classes.index(1)
    else:
        # Fallback: assume the highest label is positive
        pos_idx = np.argmax(classes)

    p_sub = float(proba[pos_idx])      # P(subscribe)
    p_not = 1.0 - p_sub                # P(not subscribe)

    if p_sub >= threshold:
        st.success(f"✅ The customer is likely to subscribe "
                   f"(P(subscribe) = {p_sub:.2f}, threshold = {threshold:.2f})")
    else:
        st.error(f"❌ The customer is unlikely to subscribe "
                 f"(P(not subscribe) = {p_not:.2f}, P(subscribe) = {p_sub:.2f}, "
                 f"threshold = {threshold:.2f})")
