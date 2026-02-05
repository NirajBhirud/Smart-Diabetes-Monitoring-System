from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import joblib
import numpy as np
from datetime import datetime
from flask import render_template
from io import BytesIO

from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet





load_dotenv()

app = Flask(__name__)

# ---------------- DATABASE CONFIG ----------------
app.config["SQLALCHEMY_DATABASE_URI"] = \
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@localhost:5432/{os.getenv('DB_NAME')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)



# =========================================================
# TABLE 1 → PREDICTION DATA (IoT readings)
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


# =========================================================
# TABLE 2 → USER REGISTRATION (Dashboard form)
# =========================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------------
# CREATE TABLES
# ---------------------------------------------------------
with app.app_context():
    db.create_all()


# =========================================================
# LOAD ML MODEL
# =========================================================
model = joblib.load("ml_model/diabetes_model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")
features = joblib.load("ml_model/feature_names.pkl")


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return "✅ Diabetes Prediction API Running (ML + DB Connected)"




@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")



@app.route("/register_page")
def register_page():
    return render_template("register.html")
# ---------------------------------------------------------
# REGISTER ROUTE  (Your Form)
# ---------------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json

        user = User(
            name=data["name"],
            age=data["age"],
            height=data["height"],
            weight=data["weight"],
            bmi=data["bmi"]
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({
            "message": "User Registered Successfully ✅",
            "user_id": user.id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ---------------------------------------------------------
# PREDICT ROUTE  (IoT → ML → DB)
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        values = np.array([[float(data[f]) for f in features]])
        values = scaler.transform(values)

        pred = model.predict(values)[0]
        prob = float(model.predict_proba(values)[0][1])
        label = "Diabetic" if pred == 1 else "Non-Diabetic"

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
# Live Data API (for Dashboard Live Panel)
# ---------------------------------------------------------
@app.route("/live")
def live_data():
    record = Prediction.query.order_by(Prediction.created_at.desc()).first()

    if not record:
        return jsonify({"error": "No data"})

    return jsonify({
        "glucose": record.glucose,
        "heart_rate": record.heart_rate,
        "activity": record.activity,
        "probability": record.probability,
        "prediction": record.prediction,
        "time": record.created_at.strftime("%H:%M:%S")
    })


# ---------------------------------------------------------
# Latest User Info (for Dashboard Patient Info)
# ---------------------------------------------------------
@app.route("/user")
def latest_user():
    user = User.query.order_by(User.created_at.desc()).first()

    if not user:
        return jsonify({"name": "--", "age": "--"})

    return jsonify({
        "name": user.name,
        "age": user.age
    })


# ---------------------------------------------------------
# REPORT DOWNLOAD ROUTE  (PDF)
# ---------------------------------------------------------
@app.route("/report")
def report():
    records = Prediction.query.order_by(Prediction.created_at.desc()).all()

    buffer = BytesIO()   # memory file

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Smart Diabetes Monitoring Report", styles['Title']))

    for r in records:
        line = f"{r.created_at} | Glucose: {r.glucose} | Risk: {round(r.probability*100,2)}%"
        elements.append(Paragraph(line, styles['Normal']))

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Diabetes_Report.pdf",
        mimetype='application/pdf'
    )



# ---------------------------------------------------------
# HISTORY ROUTE  (Dashboard Graph)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)