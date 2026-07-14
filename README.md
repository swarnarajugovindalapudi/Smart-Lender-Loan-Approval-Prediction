# Smart Lender - Loan Approval Prediction System

## Project Overview
Smart Lender is a machine learning-based web application designed to predict the eligibility of loan applicants. By leveraging historical financial data, the system provides automated, fair, and transparent recommendations on whether a loan should be approved or rejected. This project was built to demonstrate the end-to-end integration of a trained machine learning model with a web-based user interface.

## Features
- **Instant Eligibility Prediction:** Get immediate feedback on loan approval status based on input parameters.
- **Robust Machine Learning Backend:** Utilizes a Random Forest Classifier trained on balanced data for accurate predictions.
- **Strict Preprocessing Parity:** The backend strictly enforces exactly the same feature scaling, encoding, and ordering used during model training to eliminate inference bugs.
- **Client-Side Validation:** Ensures that all numerical inputs (e.g., Income, Loan Amount) and dropdowns are correctly filled before submission.
- **Responsive Web Interface:** A clean, banking-inspired layout built entirely from scratch with HTML5, CSS3, and Vanilla JavaScript (No Bootstrap or Tailwind).

## Technology Stack
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend Framework:** Python, Flask
- **Machine Learning:** Scikit-Learn, Pandas, Imbalanced-Learn (SMOTE)
- **Model Serialization:** Pickle (`pkl`)

## Folder Structure
```text
SmartLender/
│
├── app/
│   ├── app.py                     # Main Flask routing logic
│   ├── static/                    # Frontend assets
│   │   ├── css/style.css
│   │   └── js/validation.js
│   ├── templates/                 # HTML UI layouts
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── predict.html
│   │   └── result.html
│   └── utils/
│       └── model_helper.py        # ML inference business logic
│
├── dataset/
│   └── loan_prediction.csv        # Original SkillWallet Dataset
│
├── models/                        # Serialized artifacts
│   ├── best_model.pkl             # Trained Random Forest
│   ├── encoders.pkl               # Dictionary of LabelEncoders
│   ├── feature_columns.pkl        # List of feature names
│   └── scaler.pkl                 # StandardScaler
│
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb  # Data exploration and model training
│
├── tests/
│   └── test_backend.py            # Local inference test script
│
├── requirements.txt               # Required Python packages
└── README.md                      # This file
```

## Dataset Information
The project strictly uses the **SkillWallet Loan Prediction Dataset**. The data contains records of past applicants, detailing their demographic information (Gender, Education, Dependents, Marital Status) and financial history (Applicant Income, Co-applicant Income, Loan Amount, Credit History).

## ML Pipeline
1. **Data Cleaning:** Imputed missing numerical values with medians and categorical values with modes. Missing values distribution was carefully analyzed.
2. **EDA:** Generated distribution plots, correlation heatmaps, feature importance charts, and model accuracy comparisons.
3. **Encoding:** Converted categorical strings into integers using `LabelEncoder`.
4. **Balancing:** Addressed the class imbalance in loan approvals using SMOTE (Synthetic Minority Over-sampling Technique).
5. **Scaling:** Scaled numerical data using `StandardScaler` to ensure features had uniform magnitude.

## Model Selection
Multiple models were rigorously trained and evaluated:
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

For each model, a Classification Report and Confusion Matrix were generated. The best model was automatically selected based on accuracy. **Random Forest Classifier** emerged as the highest-performing model on this specific dataset partition, proving more robust than XGBoost against overfitting in this instance.

## Installation Steps
1. Ensure Python 3.8+ is installed on your machine.
2. Clone this repository to your local computer.
3. Open a terminal and navigate to the project directory:
   ```bash
   cd SmartLender
   ```
4. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application Locally
1. Navigate to the project root directory.
2. Start the Flask application as a module:
   ```bash
   python -m app.app
   ```
3. Open your web browser and go to `http://127.0.0.1:5000/`

## Deployment (Render Compatibility)
The application is configured to run effortlessly on cloud platforms like [Render](https://render.com/).
- **Requirements:** The `requirements.txt` includes `gunicorn`, a production-ready WSGI HTTP server.
- **Start Command:** When deploying to Render, set the start command to:
  ```bash
  gunicorn app.app:app
  ```
This natively bypasses the need for a `Procfile` while maintaining standard deployment practices.

## Screenshots
*(Insert screenshots of the Landing Page, Prediction Form, and Result Pages here)*

## Future Scope
- Transition the frontend to a modern component-based framework like React if scale demands it.
- Integrate a cloud database (e.g., PostgreSQL or MongoDB) to store applicant histories and feedback loops.
- Deploy the application to a cloud hosting provider (AWS, Render, or Heroku).
- Introduce Explainable AI (XAI) features like SHAP values to explain *why* an applicant was rejected.

## License
MIT License. Feel free to use and modify for educational purposes.
