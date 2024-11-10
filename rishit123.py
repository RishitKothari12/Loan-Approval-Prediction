import streamlit as st
import pandas as pd
import pickle
import gzip

# Load your trained model and any other necessary data files
@st.cache_resource
def load_model():
    with gzip.open("data.pkl.gz", "rb") as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model()

# Load training and testing data if needed
@st.cache_data
def load_data():
    train_data = pd.read_csv("Loan_Approval_Data_Train.csv")
    test_data = pd.read_csv("Loan_Approval_Data_Test.csv")
    return train_data, test_data

train_data, test_data = load_data()

# Set up the Streamlit page
st.title("Loan Approval Prediction")
st.write("Predict loan approval based on applicant details.")

# Sidebar for user inputs
st.sidebar.header("Applicant Details")
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term (months)", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", options=[0, 1])
property_area = st.sidebar.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

# Add more fields as required for your model inputs...

# Predict function
def predict_approval(model, input_data):
    prediction = model.predict(input_data)
    return "Approved" if prediction == 1 else "Rejected"

# Create a button for prediction
if st.button("Predict Loan Approval"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    # Make prediction
    prediction = predict_approval(model, input_data)
    st.write(f"Loan Approval Status: {prediction}")

