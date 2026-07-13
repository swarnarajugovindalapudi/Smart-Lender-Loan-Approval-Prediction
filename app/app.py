from flask import Flask, render_template, request
from utils.model_helper import predictor
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Render the landing page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle loan prediction requests."""
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        try:
            # Extract all form data as a dictionary
            form_data = request.form.to_dict()
            
            # Predict using the helper module
            is_approved = predictor.predict(form_data)
            
            return render_template('result.html', is_approved=is_approved)
            
        except Exception as e:
            print(f"Prediction Error: {traceback.format_exc()}")
            return render_template('result.html', error=str(e))

if __name__ == '__main__':
    # Run in debug mode for local development
    app.run(debug=True, port=5000)
