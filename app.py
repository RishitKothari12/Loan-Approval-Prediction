from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Define mappings for categorical variables based on the training preprocessing
loan_grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
cb_default_mapping = {'Y': 1, 'N': 0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect numerical and encoded inputs
    person_age = float(request.form['person_age'])
    person_income = float(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_grade = loan_grade_mapping[request.form['loan_grade']]  # Map loan grade input
    loan_amnt = float(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    loan_percent_income = float(request.form['loan_percent_income'])

    cb_person_default_on_file = cb_default_mapping.get(request.form['cb_person_default_on_file'], None)
    if cb_person_default_on_file is None:
      return "Error: Invalid value for credit bureau default (Y or N required)"

    #cb_person_default_on_file = cb_default_mapping[request.form['cb_person_default_on_file']]
    cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])

    # One-Hot Encoding for person_home_ownership
    person_home_ownership_rent = 1 if request.form['person_home_ownership'] == 'RENT' else 0
    person_home_ownership_mortgage = 1 if request.form['person_home_ownership'] == 'MORTGAGE' else 0
    person_home_ownership_own = 1 if request.form['person_home_ownership'] == 'OWN' else 0
    person_home_ownership_other = 1 if request.form['person_home_ownership'] == 'OTHER' else 0

    # One-Hot Encoding for loan_intent
    loan_intent_education = 1 if request.form['loan_intent'] == 'EDUCATION' else 0
    loan_intent_medical = 1 if request.form['loan_intent'] == 'MEDICAL' else 0
    loan_intent_personal = 1 if request.form['loan_intent'] == 'PERSONAL' else 0
    loan_intent_venture = 1 if request.form['loan_intent'] == 'VENTURE' else 0
    loan_intent_debtconsolidation = 1 if request.form['loan_intent'] == 'DEBTCONSOLIDATION' else 0
    loan_intent_homeimprovement = 1 if request.form['loan_intent'] == 'HOMEIMPROVEMENT' else 0

    # Prepare the feature array in the same order as the training data
    features = [
        person_age, person_income, person_emp_length, loan_grade, loan_amnt,
        loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length,
        person_home_ownership_rent, person_home_ownership_mortgage, person_home_ownership_own, person_home_ownership_other,
        loan_intent_education, loan_intent_medical, loan_intent_personal, loan_intent_venture,
        loan_intent_debtconsolidation, loan_intent_homeimprovement
    ]

    # Make the prediction
    prediction = model.predict([features])[0]
    result = "Approved" if prediction == 1 else "Denied"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)