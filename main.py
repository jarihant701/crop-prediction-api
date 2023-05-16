from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Define the names of the crops
crop_names = [
    "arhar",
    "barley",
    "gram",
    "guarseed",
    "Cotton",
    "maize",
    "moong",
    "peasandbeans",
    "Rice",
    "Rapeseed",
    "sesamum",
    "sugarcane",
    "Urad",
    "wheat",
]

# Define the features used by each model
model_features = {
    "arhar": ["SlAc%", "N%", "MlAl%", "SlAl%"],
    "barley": ["Cu%", "MlAl%", "SlAl%"],
    "Cotton": ["Cu%", "MlAl%", "SlAl%"],
    "gram": ["Cu%", "MlAl%", "SlAl%"],
    "guarseed": ["Cu%", "MlAl%", "SlAl%"],
    "maize": ["SlAc%", "N%", "MlAl%", "SlAl%"],
    "moong": ["Cu%", "MlAl%", "SlAl%"],
    "peasandbeans": ["SlAc%", "N%", "MlAl%", "SlAl%"],
    "Rice": ["Cu%", "MlAl%", "SlAl%"],
    "sesamum": ["Cu%", "MlAl%", "SlAl%"],
    "sugarcane": ["SlAc%", "N%", "MlAl%", "SlAl%"],
    "Urad": ["SlAc%", "N%", "MlAl%", "SlAl%"],
    "wheat": ["MlAl%", "SlAl%"],
    "Rapeseed": ["Cu%", "MlAl%", "SlAl%"],
}

# Load the trained machine learning models for all crops
KNR_models = {
    crop: joblib.load(f"{crop}/{crop}_yield_prediction_KNR_model3.joblib")
    for crop in crop_names
}
SVR_models = {
    crop: joblib.load(f"{crop}/{crop}_yield_prediction_SVR_model2.joblib")
    for crop in crop_names
}


@app.route("/", methods=["GET"])
def server_status():
    return jsonify({"status": "Server running"})


# Define the route for the API
@app.route("/predict", methods=["POST"])
def predict_yield():
    # Get the JSON data from the POST request
    data = request.get_json(force=True)

    predictions = {}
    for crop, model in KNR_models.items():
        # Extract the nutrient values for this model
        nutrients = [data[feature] for feature in model_features[crop]]

        # Create a numpy array with these values
        input_data = np.array([nutrients])

        # Use the model to make a prediction and store it in the predictions dictionary
        predictions[f"{crop}_KNR"] = model.predict(input_data)[0]

    for crop, model in SVR_models.items():
        # Extract the nutrient values for this model
        nutrients = [data[feature] for feature in model_features[crop]]

        # Create a numpy array with these values
        input_data = np.array([nutrients])

        # Use the model to make a prediction and store it in the predictions dictionary
        predictions[f"{crop}_SVR"] = model.predict(input_data)[0]
    # Return the predictions as a JSON response
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=True)
