# =========================================================
# SMART DIABETES MONITORING SYSTEM
# FINAL PRODUCTION FLASK + POSTGRESQL + ML API
# IoT → Cloud → ML → DB → Dashboard
# =========================================================

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import joblib
import numpy as np
from datetime import datetime

# ---------------------------------------------------------
# LOAD ENV VARIABLES  (VERY IMPORTANT - SECURITY)
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# App Init
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# DATABASE CONFIG (Secure)
# ---------------------------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = \
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@localhost:5432/{os.getenv('DB_NAME')}"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================================================
# DATABASE TABLE
# =========================================================
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    glucose = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    activity = db.Column(db.Integer, nullable=False)

    prediction = db.Column(db.String(20))
    probability = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------------
# CREATE TABLES
# ---------------------------------------------------------
with app.app_context():
    db.create_all()


# =========================================================
# LOAD ML MODEL (once at startup)
# =========================================================
print("Loading ML model...")

try:
    model = joblib.load("ml_model/diabetes_model.pkl")
    scaler = joblib.load("ml_model/scaler.pkl")
    features = joblib.load("ml_model/feature_names.pkl")
    print("✅ ML model loaded successfully")
except Exception as e:
    print("❌ ML loading failed:", e)


# =========================================================
# ROUTES
# =========================================================

# ---------------------------------------------------------
# Home Route
# ---------------------------------------------------------
@app.route("/")
def home():
    return "✅ Diabetes Prediction API Running (ML + DB Connected)"


# ---------------------------------------------------------
# Predict Route  (IoT → Cloud → ML → DB)
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json

        # Validate input
        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Prepare input
        values = np.array([[float(data[f]) for f in features]])
        values = scaler.transform(values)

        # Predict
        pred = model.predict(values)[0]
        prob = float(model.predict_proba(values)[0][1])

        label = "Diabetic" if pred == 1 else "Non-Diabetic"

        # Save to DB
        record = Prediction(
            glucose=data["Glucose"],
            age=data["Age"],
            bmi=data["BMI"],
            heart_rate=data["HeartRate"],
            activity=data["Activity"],
            prediction=label,
            probability=prob
        )

        db.session.add(record)
        db.session.commit()

        return jsonify({
            "prediction": label,
            "probability": round(prob, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# History Route  (Dashboard)
# ---------------------------------------------------------
@app.route("/history")
def history():

    records = Prediction.query.order_by(Prediction.created_at.desc()).all()

    result = [
        {
            "glucose": r.glucose,
            "age": r.age,
            "bmi": r.bmi,
            "heart_rate": r.heart_rate,
            "activity": r.activity,
            "prediction": r.prediction,
            "probability": r.probability,
            "time": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in records
    ]

    return jsonify(result)


# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
