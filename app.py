import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model (Suppressing print statements for brevity)
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    model = None # Set model to None if loading fails

# Feature order is mandatory: ['weight', 'age', 'shoe_size', 'arm_length', 'leg_length']
REQUIRED_FEATURES = ['weight', 'age', 'shoe_size', 'arm_length', 'leg_length']

@app.route('/')
def home():
    return "Height Prediction API: POST data to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load.'}), 500

    try:
        data = request.get_json(force=True)
        input_data = [data.get(f) for f in REQUIRED_FEATURES]

        # Validation check for numeric data
        if any(item is None or not isinstance(item, (int, float)) for item in input_data):
            return jsonify({'error': 'Invalid or missing numeric data.'}), 400

        # Prediction logic
        final_features = np.array([input_data])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return jsonify({'predicted_height': output, 'unit': 'units'})

    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=False)