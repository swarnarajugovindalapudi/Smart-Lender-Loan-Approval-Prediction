import os
import sys

# Add the 'app' directory to the system path so 'utils' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from utils.model_helper import predictor

def run_tests():
    print("Running Backend Inference Test from tests directory...")
    
    # Simulate a user filling out the HTML form with safe data
    mock_form_data = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': '5000',
        'CoapplicantIncome': '2000',
        'LoanAmount': '150',
        'Loan_Amount_Term': '360',
        'Credit_History': '1.0',
        'Property_Area': 'Urban'
    }

    try:
        is_approved = predictor.predict(mock_form_data)
        print(f"Prediction successful! Loan Approved: {is_approved}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    run_tests()
