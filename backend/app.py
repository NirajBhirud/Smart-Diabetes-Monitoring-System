from flask import Flask, request, jsonify, render_template, make_response
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import joblib
import numpy as np
from datetime import datetime

# ---------------- LOAD ENV ----------------
load_dotenv()

app = Flask(__name__)

# ---------------- DATABASE CONFIG ----------------
app.config["SQLALCHEMY_DATABASE_URI"] = \
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@localhost:5432/{os.getenv('DB_NAME')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# =========================================================
# TABLE 1 → PREDICTION DATA
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
# TABLE 2 → AUTH TABLE
# =========================================================
class Auth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# =========================================================
# TABLE 3 → USER REGISTRATION TABLE
# =========================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(120))
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# CREATE TABLES
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
def landing():
    return render_template("landing.html")

@app.route("/auth")
def auth():
    return render_template("auth.html")

@app.route("/register_page")
def register_page():
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# =========================================================
# SIGNUP
# =========================================================
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json

    user = Auth(
        email=data["email"],
        password=data["password"]
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Signup successful"})


# =========================================================
# LOGIN
# =========================================================
@app.route("/login", methods=["POST"])
def login():
    data = request.json

    user = Auth.query.filter_by(
        email=data["email"],
        password=data["password"]
    ).first()

    if user:
        return jsonify({"message": "Login success"})
    else:
        return jsonify({"error": "Invalid credentials"})

# =========================================================
# GET LAST AUTH USER (For Auto-fill Register Page)
# =========================================================
@app.route("/get_auth_user")
def get_auth_user():
    user = Auth.query.order_by(Auth.created_at.desc()).first()

    if not user:
        return jsonify({"username": "", "email": ""})

    return jsonify({
        "username": user.username,
        "email": user.email
    })

# =========================================================
# REGISTER USER (From Register Page)
# =========================================================
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json

        user = User(
            username=data["username"],
            email=data["email"],
            name=data["name"],
            age=data["age"],
            height=data["height"],
            weight=data["weight"],
            bmi=data["bmi"]
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({
            "message": "User Registered Successfully",
            "user_id": user.id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# PREDICT
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
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

# =========================================================
# LIVE DATA
# =========================================================
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

# =========================================================
# USER INFO
# =========================================================
@app.route("/user")
def latest_user():
    user = User.query.order_by(User.created_at.desc()).first()

    if not user:
        return jsonify({"name": "--", "age": "--"})

    return jsonify({
        "name": user.name,
        "age": user.age
    })

# =========================================================
# HISTORY
# =========================================================
@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.created_at.desc()).all()

    result = [
        {
            "glucose": r.glucose,
            "heart_rate": r.heart_rate,
            "activity": r.activity,
            "probability": r.probability,
            "time": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in records
    ]

    return jsonify(result)

# =========================================================
# REPORT DOWNLOAD
# =========================================================
@app.route("/report")
def report():
    user = User.query.order_by(User.created_at.desc()).first()
    p = Prediction.query.order_by(Prediction.created_at.desc()).first()

    if not user or not p:
        return "No data available"

    return render_template(
        "report.html",
        user=user,
        p=p,
        now=datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        report_id=int(datetime.now().timestamp())
    )


# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
