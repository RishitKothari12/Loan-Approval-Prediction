# rishit123.py (or your app's main script)
import gzip
import pickle
import streamlit as st

# Function to load the model
def load_model():
    with gzip.open('data.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model when the script starts
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

# Prediction function
def predict_approval(model, input_data):
    prediction = model.predict(input_data)  # Predict using the model
    return prediction

# Streamlit app to handle input and display the result
def app():
    input_data = st.text_input("Enter your input data here")  # or use another input method
    if input_data:
        # Format or preprocess the input data if needed before passing to the model
        prediction = predict_approval(model, input_data)  # Call the predict function
        st.write(f"Prediction: {prediction}")

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
    
    # Make prediction
    prediction = predict_approval(model, input_data)
    st.write(f"Loan Approval Status: {prediction}")
