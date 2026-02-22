from flask import session
from flask import Flask, request, jsonify, render_template, make_response
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import joblib
import numpy as np
from datetime import datetime
from flask import redirect
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

    steps = db.Column(db.Integer, nullable=False)      # 👈 NEW
    activity = db.Column(db.Integer, nullable=False)   # 0/1/2 (ML)

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
print("APP scaler features:", scaler.n_features_in_)
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

    # 🔥 AUTO LOGIN (create session)
    user = User.query.filter_by(email=data["email"]).first()
    if user:
        session["user_id"] = user.id

    return jsonify({"message": "Signup & login successful"})


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

    user = db.session.get(User, session["user_id"])
    return jsonify({
        "name": user.name,
        "age": user.age
    })
def map_steps_to_activity(steps):
    """
    Convert raw step count to activity level
    Must match training distribution: {0,1,2}
    """
    if steps < 5000:
        return 0   # Sedentary
    elif steps < 10000:
        return 1   # Lightly active
    else:
        return 2   # Moderately active
# ---- Clinical constants (MUST MATCH TRAINING) ----
BASE_WEIGHTS = np.array([5.0, 1.2, 1.5, 0.9, 0.7, 4.0, 6.0, 8.0])
ZONE_MULT    = {0: 1.0, 1: 1.5, 2: 3.0}

GLUCOSE_IDX = 0   # "Glucose"
ZONE_IDX    = 5   # "Glucose_Zone"

def apply_weights(X):
    Xw = X.copy().astype(float) * BASE_WEIGHTS
    zones = X[:, ZONE_IDX].astype(int)
    Xw[:, GLUCOSE_IDX] *= np.array([ZONE_MULT[z] for z in zones])
    return Xw
# ==============================
# PREDICT (AGE & BMI FROM DB)
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = db.session.get(User, session["user_id"])
    data = request.json or {}

    try:
        glucose = float(data.get("Glucose"))
        heart_rate = float(data.get("HeartRate"))
        steps = int(data.get("Activity"))   # steps from frontend
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing input data"}), 400
    activity = map_steps_to_activity(steps)
    # ---- SAME FEATURE ENGINEERING AS TRAINING ----
    if glucose <= 99:
        zone = 0
    elif glucose <= 125:
        zone = 1
    else:
        zone = 2

    excess = max(0, glucose - 125)
    is_diab = 1 if glucose >= 126 else 0

    values = np.array([[
        glucose,
        user.age,
        user.bmi,
        heart_rate,
        activity,
        zone,
        excess,
        is_diab
    ]])

    # Debug check (optional)
    print("Input shape:", values.shape)
    print("Scaler expects:", scaler.n_features_in_)

    values_weighted = apply_weights(values)
    values_scaled   = scaler.transform(values_weighted)

    pred = model.predict(values_scaled)[0]
    prob = float(model.predict_proba(values_scaled)[0][1])

    label = "Diabetic" if pred == 1 else "Non-Diabetic"

    record = Prediction(
        user_id=user.id,
        glucose=glucose,
        heart_rate=heart_rate,
        steps=steps,            # ✅ ADD THIS
        activity=activity,
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
        "steps": record.steps,          # ✅ REAL STEPS
        "activity_level": record.activity,  # optional (debug/admin)
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
    ).order_by(Prediction.created_at.asc()).all()

    return jsonify([
        {
            "glucose": r.glucose,
            "heart_rate": r.heart_rate,
            "steps": r.steps,          # ✅ ADD THIS
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

    user = db.session.get(User, session["user_id"])
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