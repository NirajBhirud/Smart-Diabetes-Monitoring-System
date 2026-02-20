from flask import session
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
app.secret_key = "smart-diabetes-secret"

# ---------------- DATABASE CONFIG ----------------
app.config["SQLALCHEMY_DATABASE_URI"] = \
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@localhost:5432/{os.getenv('DB_NAME')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# =========================================================
# TABLE 1 → PREDICTION DATA
# =========================================================
class Auth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    glucose = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    activity = db.Column(db.Integer, nullable=False)

    prediction = db.Column(db.String(20))
    probability = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# ==============================
# LOAD ML MODEL
# ==============================
model = joblib.load("ml_model/diabetes_model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")

# ==============================
# ROUTES
# ==============================

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
    if "user_id" not in session:
        return redirect("/auth")
    return render_template("dashboard.html")


# ==============================
# SIGNUP
# ==============================
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json

    if Auth.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already exists"}), 409

    auth = Auth(
        email=data["email"],
        password=data["password"]
    )
    db.session.add(auth)
    db.session.commit()

    return jsonify({"message": "Signup successful"})


# ==============================
# REGISTER (CREATE USER PROFILE)
# ==============================
@app.route("/register", methods=["POST"])
def register():
    data = request.json

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already registered"}), 409

    user = User(
        email=data["email"],
        name=data["name"],
        age=data["age"],
        gender=data.get("gender"),
        height=data["height"],
        weight=data["weight"],
        bmi=data["bmi"]
    )

    db.session.add(user)
    db.session.commit()

    # Auto login after registration
    session["user_id"] = user.id

    return jsonify({"message": "User registered successfully"})


# ==============================
# LOGIN
# ==============================
@app.route("/login", methods=["POST"])
def login():
    data = request.json

    auth = Auth.query.filter_by(
        email=data["email"],
        password=data["password"]
    ).first()

    if not auth:
        return jsonify({"error": "Invalid credentials"}), 401

    user = User.query.filter_by(email=auth.email).first()
    if not user:
        return jsonify({"error": "User profile not found"}), 404

    session["user_id"] = user.id
    return jsonify({"message": "Login success"})


# ==============================
# LOGOUT
# ==============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/auth")


# ==============================
# USER INFO (DASHBOARD)
# ==============================
@app.route("/user")
def user_info():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.get(session["user_id"])
    return jsonify({
        "name": user.name,
        "age": user.age
    })


# ==============================
# PREDICT (AGE & BMI FROM DB)
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.get(session["user_id"])
    data = request.json

    values = np.array([[
        float(data["Glucose"]),
        user.age,
        user.bmi,
        float(data["HeartRate"]),
        int(data["Activity"])
    ]])

    values = scaler.transform(values)

    pred = model.predict(values)[0]
    prob = float(model.predict_proba(values)[0][1])
    label = "Diabetic" if pred == 1 else "Non-Diabetic"

    record = Prediction(
        user_id=user.id,
        glucose=data["Glucose"],
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


# ==============================
# LIVE DATA (USER-SPECIFIC)
# ==============================
@app.route("/live")
def live():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    record = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.desc()).first()

    if not record:
        return jsonify({"error": "No data"})

    return jsonify({
        "glucose": record.glucose,
        "heart_rate": record.heart_rate,
        "activity": record.activity,
        "probability": record.probability,
        "prediction": record.prediction
    })


# ==============================
# HISTORY
# ==============================
@app.route("/history")
def history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    records = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.desc()).all()

    return jsonify([
        {
            "glucose": r.glucose,
            "heart_rate": r.heart_rate,
            "activity": r.activity,
            "probability": r.probability,
            "time": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        } for r in records
    ])


# ==============================
# REPORT
# ==============================
@app.route("/report")
def report():
    if "user_id" not in session:
        return redirect("/auth")

    user = User.query.get(session["user_id"])
    p = Prediction.query.filter_by(
        user_id=user.id
    ).order_by(Prediction.created_at.desc()).first()

    if not p:
        return "No data available"

    return render_template(
        "report.html",
        user=user,
        p=p,
        now=datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        report_id=int(datetime.now().timestamp())
    )


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)