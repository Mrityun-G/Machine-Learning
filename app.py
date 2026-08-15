import numpy as np
from flask import request, jsonify, Flask
import joblib

app = Flask(__name__)

# Load the trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')

# Load the trained StandardScaler object
scaler = joblib.load('scaler.joblib')

print("Model and scaler loaded successfully.")

from typing import Dict, List
import numpy as np
from flask import request, jsonify
import logging

# Extract feature order to a constant for better maintainability
FEATURE_ORDER = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Define a function to extract input features from the request data
def extract_input_features(data: Dict) -> List:
    try:
        # Use a list comprehension to extract features in the correct order
        return [data[feature] for feature in FEATURE_ORDER]
    except KeyError as e:
        # Raise a custom error with a descriptive message
        raise ValueError(f"Missing feature in input data: {e}. Expected features: {FEATURE_ORDER}")

# Define a function to preprocess the input features
def preprocess_input_features(input_features: List) -> np.ndarray:
    # Convert the list to a NumPy array
    input_array = np.array(input_features)
    # Reshape the input data to a 2D array
    input_array_reshaped = input_array.reshape(1, -1)
    # Use the loaded scaler object to transform the input features
    return scaler.transform(input_array_reshaped)

# Define a function to make a prediction using the preprocessed input features
def make_prediction(scaled_input: np.ndarray) -> int:
    # Use the loaded model object to make a prediction
    prediction = model.predict(scaled_input)
    # Convert the prediction to a Python int
    return int(prediction[0])

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
async def predict():
    try:
        # Get the request data
        data = request.get_json(force=True)
        logging.info("Received data for prediction: %s", data)
        
        # Extract the input features
        input_features = extract_input_features(data)
        
        # Preprocess the input features
        scaled_input = preprocess_input_features(input_features)
        
        # Make a prediction
        result = make_prediction(scaled_input)
        
        # Return the prediction result
        return jsonify({'prediction': result}), 200
    except Exception as e:
        # Log the error and return a 500 error response
        logging.error("Error making prediction: %s", e)
        return jsonify({"error": "Internal Server Error"}), 500

print("Prediction endpoint updated with preprocessing and prediction logic.")
if __name__ == "__main__":
    print("Starting prediction API with preprocessing and model inference...")
    app.run(debug=True)
