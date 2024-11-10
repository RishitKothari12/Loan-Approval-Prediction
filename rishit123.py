import streamlit as st
import pandas as pd
import pickle
import gzip

# Load your trained model
@st.cache_resource
def load_model():
    try:
        with gzip.open("data.pkl.gz", "rb") as f:
            model_data = pickle.load(f)
        
        # Check if the loaded object is a dictionary and extract the model
        if isinstance(model_data, dict):
            model = model_data.get("model")  # Adjust this key based on actual structure
        else:
            model = model_data

        # Check if model has 'predict' attribute
        if not hasattr(model, "predict"):
            st.error("Loaded object is not a model with a 'predict' method.")
            return None
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Set up the Streamlit page
st.title("Loan Approval Prediction")
st.write("Predict loan approval based on applicant details.")

# Sidebar for user inputs
st.sidebar.header("Applicant Details")
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100)
person_income = st.sidebar.number_input("Annual Income", min_value=0)
person_home_ownership = st.sidebar.selectbox("Home Ownership", options=["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0)
loan_intent = st.sidebar.selectbox("Loan Intent", options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
loan_percent_income = st.sidebar.number_input("Loan Percent of Income", min_value=0.0, max_value=100.0)
cb_person_default_on_file = st.sidebar.selectbox("Default on File", options=["Y", "N"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0)

# Predict function
def predict_approval(model, input_data):
    try:
        prediction = model.predict(input_data)
        return "Approved" if prediction == 1 else "Rejected"
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Check if model loaded successfully
if model:
    # Create a button for prediction
    if st.button("Predict Loan Approval"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'person_age': [person_age],
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership],
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_default_on_file': [cb_person_default_on_file],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length]
        })
        
        # Debug: Check input data
        st.write("Input Data for Prediction:")
        st.write(input_data)
        
        # Make prediction
        prediction = predict_approval(model, input_data)
        
        # Display prediction result
        if prediction:
            st.write(f"Loan Approval Status: {prediction}")
else:
    st.error("Model could not be loaded. Please check the model file.")
