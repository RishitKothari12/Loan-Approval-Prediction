Loan Approval Prediction

Project Overview

This project focuses on predicting loan approval status using Machine Learning. By analyzing applicant details such as income, employment history, credit history, and loan parameters, the model classifies whether a loan should be approved or not. The goal is to assist financial institutions in making data-driven, consistent, and faster decisions while minimizing risks.

Dataset

The dataset contains applicant and loan-related features including applicant age, annual income, home ownership status, employment length, loan purpose, assigned loan grade, loan amount, loan interest rate, loan amount as a percentage of income, past default history, and credit history length. The target variable is loan_status which indicates whether the loan is approved (1) or not approved (0). The training dataset is stored in Loan_Approval_Data_Train.csv and the testing dataset is stored in Loan_Approval_Data_Test.csv.

Tech Stack

The project is implemented in Python using libraries such as Pandas, NumPy, and Scikit-learn for data preprocessing and model training. Flask is used as the web framework to deploy the model. The trained model is serialized using Pickle and stored in data.pkl.gz. The frontend is developed using HTML and CSS to allow user interaction with the model through a simple web interface.

Workflow

The workflow begins with data preprocessing where missing values are handled, categorical variables are encoded, and numerical features are normalized. Multiple machine learning models including Logistic Regression, Decision Trees, and Random Forest were tested, and the best performing model was selected based on evaluation metrics such as accuracy and precision. The trained model is then integrated into a Flask application where user details are entered through a web form and the model predicts loan approval status in real time.

Results

The model achieved high accuracy in predicting loan approval status. The Flask web application provides real-time predictions with an easy-to-use interface for entering applicant details and displaying whether the loan is likely to be approved.

Future Improvements

Planned improvements include deploying the application on cloud platforms such as Heroku, AWS, or Render for public access, experimenting with more advanced machine learning models such as XGBoost or LightGBM, improving the frontend with a more professional user interface, and eventually integrating the solution with a real-time loan management system.

License

Copyright © 2025 Rishit Kothari. All Rights Reserved.
Unauthorized copying, modification, distribution, or use of this project in any form is strictly prohibited without explicit permission from the author.
