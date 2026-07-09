import os
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-development-key')

@app.route('/')
def home():
    # Placeholder route for initial workspace verification
    return "Smart Lender workspace initialized successfully!"

if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=True)
