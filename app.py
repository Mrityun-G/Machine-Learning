import numpy as np
from flask import request, jsonify, Flask
import joblib

app = Flask(__name__)

# Load the trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')

# Load the trained StandardScaler object
scaler = joblib.load('scaler.joblib')

print("Model and scaler loaded successfully.")

from flask import request, jsonify
from flask import current_app
import numpy as np

FEATURE_ORDER = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

async def extract_input_features(data):
    try:
        # Convert the incoming JSON data to a list of values
        input_features = [data[feature] for feature in FEATURE_ORDER]
        return input_features
    except KeyError as e:
        raise ValueError(f"Missing feature in input data: {e}. Expected features: {FEATURE_ORDER}")

async def preprocess_input_features(input_features):
    # Convert the list to a NumPy array
    input_array = np.array(input_features)
    # Reshape the input data to a 2D array, as scaler.transform expects 2D input
    input_array_reshaped = input_array.reshape(1, -1)
    # Use the loaded scaler object to transform (preprocess) the extracted input features
    scaled_input = current_app.config['scaler'].transform(input_array_reshaped)
    return scaled_input

async def make_prediction(scaled_input):
    # Use the loaded model object to make a prediction
    prediction = current_app.config['model'].predict(scaled_input)
    # The prediction will be a NumPy array, convert it to a Python int
    result = int(prediction[0])
    return result

@app.route('/predict', methods=['POST'])
async def predict():
    try:
        data = request.get_json(force=True)
        input_features = await extract_input_features(data)
        scaled_input = await preprocess_input_features(input_features)
        result = await make_prediction(scaled_input)
        return jsonify({'prediction': result}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Starting prediction API with preprocessing and model inference...")
    app.run(debug=True)
