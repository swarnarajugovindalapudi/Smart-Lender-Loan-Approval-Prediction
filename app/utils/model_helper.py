import os
import joblib
import pandas as pd
import warnings

# Suppress sklearn warnings about feature names if they occur
warnings.filterwarnings('ignore', category=UserWarning)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODERS_PATH = os.path.join(MODELS_DIR, 'encoders.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_columns.pkl')

class LoanPredictor:
    def __init__(self):
        """Initialize and load all ML artifacts into memory."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.encoders = joblib.load(ENCODERS_PATH)
            self.feature_columns = joblib.load(FEATURES_PATH)
        except Exception as e:
            print(f"Error loading models. Ensure Milestone 1 has been completed successfully. Details: {e}")
            raise

    def predict(self, form_data: dict) -> bool:
        """
        Process the raw form data and return a boolean indicating loan approval.
        
        Args:
            form_data (dict): A dictionary containing raw inputs from the HTML form.
                              Expected keys must align with self.feature_columns.
        
        Returns:
            bool: True if loan is approved, False otherwise.
        """
        # 1. Construct a dictionary mapped strictly to the feature_columns to guarantee order
        processed_data = {}
        
        for col in self.feature_columns:
            # Safely extract the value, default to 0 or '' if somehow missing (though frontend should catch this)
            raw_value = form_data.get(col, '')
            
            if col in self.encoders:
                # It's a categorical column
                le = self.encoders[col]
                # If the raw_value is unseen (which shouldn't happen with strict dropdowns), 
                # we handle it gracefully by picking the first class to avoid hard crashes in a student project.
                if raw_value not in le.classes_:
                    print(f"Warning: Unseen label '{raw_value}' for column '{col}'. Using fallback.")
                    raw_value = le.classes_[0]
                
                # Transform using the exact label encoder from training
                processed_data[col] = le.transform([str(raw_value)])[0]
            else:
                # It's a numerical column, ensure it's a float
                try:
                    processed_data[col] = float(raw_value)
                except ValueError:
                    processed_data[col] = 0.0

        # 2. Convert to DataFrame to ensure exact structure for scaler/model
        df_input = pd.DataFrame([processed_data], columns=self.feature_columns)
        
        # 3. Scale numerical features (StandardScaler scales the entire df in this workflow)
        X_scaled = self.scaler.transform(df_input)
        
        # 4. Predict
        prediction = self.model.predict(X_scaled)
        
        # In training, the target was LabelEncoded. Usually: 0 -> N, 1 -> Y
        # We return True for Approval (1), False for Rejection (0)
        return bool(prediction[0] == 1)

# Initialize a global instance so it's loaded once when the app starts
predictor = LoanPredictor()
