import numpy as np
from flask import request, jsonify, Flask
import joblib

app = Flask(__name__)

# Load the trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')

# Load the trained StandardScaler object
scaler = joblib.load('scaler.joblib')

print("Model and scaler loaded successfully.")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data for prediction:", data)

    # Convert the incoming JSON data to a list of values
    # Assuming 'data' is a dictionary with feature names as keys
    # The order of features should match the training data
    # For simplicity, we'll assume a fixed order based on the original DataFrame columns, excluding 'Outcome'
    feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    try:
        input_features = [data[feature] for feature in feature_order]
    except KeyError as e:
        return jsonify({"error": f"Missing feature in input data: {e}. Expected features: {feature_order}"}), 400

    # Convert the list to a NumPy array
    input_array = np.array(input_features)

    # Reshape the input data to a 2D array, as scaler.transform expects 2D input
    input_array_reshaped = input_array.reshape(1, -1)

    # Use the loaded scaler object to transform (preprocess) the extracted input features
    scaled_input = scaler.transform(input_array_reshaped)

    # Use the loaded model object to make a prediction
    prediction = model.predict(scaled_input)

    # The prediction will be a NumPy array, convert it to a Python int
    result = int(prediction[0])

    return jsonify({'prediction': result}), 200

print("Prediction endpoint updated with preprocessing and prediction logic.")

if __name__ == "__main__":
    print("Starting prediction API with preprocessing and model inference...")
    app.run(debug=True)
