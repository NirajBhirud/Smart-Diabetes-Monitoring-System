# =====================================
# Flask Cloud API for Diabetes Prediction
# =====================================

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# -------------------------------------
# Load ML files once (server start)
# -------------------------------------
print("Loading ML model...")

try:
    model = joblib.load("ml_model/diabetes_model.pkl")
    scaler = joblib.load("ml_model/scaler.pkl")
    features = joblib.load("ml_model/feature_names.pkl")
    print("Model loaded successfully ✅")
except Exception as e:
    print("Error loading model or scaler:", e)


# -------------------------------------
# Home route
# -------------------------------------
@app.route("/")
def home():
    return "Diabetes Prediction API Running ✅"


# -------------------------------------
# Prediction route
# -------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    try:
        values = [data[f] for f in features]

        values = np.array(values).reshape(1, -1)

        values = scaler.transform(values)

        prediction = model.predict(values)[0]
        probability = model.predict_proba(values)[0][1]

        result = {
            "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "probability": round(float(probability), 3)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------------
# Run server
# -------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
