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
# MODELS
# =========================================================
class Auth(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    name       = db.Column(db.String(100), nullable=False)
    age        = db.Column(db.Integer, nullable=False)
    gender     = db.Column(db.String(10))
    contact    = db.Column(db.String(15))
    height     = db.Column(db.Float)
    weight     = db.Column(db.Float)
    bmi        = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    glucose     = db.Column(db.Float, nullable=False)
    heart_rate  = db.Column(db.Float, nullable=False)
    spo2        = db.Column(db.Float)
    steps       = db.Column(db.Integer, nullable=False)
    activity    = db.Column(db.Integer, nullable=False)
    prediction  = db.Column(db.String(20))
    probability = db.Column(db.Float)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# =========================================================
# LOAD ML MODEL
# =========================================================
model  = joblib.load("ml_model/diabetes_model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")
print("Scaler features:", scaler.n_features_in_)


# =========================================================
# ✅ FIX — ACTIVE IOT USER TRACKER
# =========================================================
# This is the ROOT CAUSE of the bug:
#   ESP32 hardcoded user_id=1, so every reading went to user 1.
#   Any other logged-in user's dashboard got "No data".
#
# Solution: one server-side variable tracks whoever logged in last.
# On login/register → update active_iot_user_id.
# ESP32's /iot/predict → saves to active_iot_user_id.
# Every user's /live → fetches their own session user_id (unchanged).
# Result: ESP32 data always goes to the currently active user.
# =========================================================
active_iot_user_id = 1   # default, updated on every login/register


# =========================================================
# CLINICAL CONSTANTS (must match training)
# =========================================================
BASE_WEIGHTS = np.array([5.0, 1.2, 1.5, 0.9, 0.7, 4.0, 6.0, 8.0])
ZONE_MULT    = {0: 1.0, 1: 1.5, 2: 3.0}
GLUCOSE_IDX  = 0   # "Glucose"
ZONE_IDX     = 5   # "Glucose_Zone"


def apply_weights(X):
    Xw    = X.copy().astype(float) * BASE_WEIGHTS
    zones = X[:, ZONE_IDX].astype(int)
    Xw[:, GLUCOSE_IDX] *= np.array([ZONE_MULT[z] for z in zones])
    return Xw


def map_steps_to_activity(steps):
    if steps < 5000:  return 0
    if steps < 10000: return 1
    return 2


def build_glucose_features(glucose):
    """ADA zone engineering — identical to training."""
    if   glucose <= 99:  zone = 0
    elif glucose <= 125: zone = 1
    else:                zone = 2
    excess  = max(0, glucose - 125)
    is_diab = 1 if glucose >= 126 else 0
    return zone, excess, is_diab


def run_prediction(glucose, age, bmi, heart_rate, activity):
    """Single prediction pipeline used by both /predict and /iot/predict."""
    zone, excess, is_diab = build_glucose_features(glucose)
    values = np.array([[
        glucose, age, bmi, heart_rate, activity,
        zone, excess, is_diab
    ]])
    values_weighted = apply_weights(values)
    values_scaled   = scaler.transform(values_weighted)
    pred  = model.predict(values_scaled)[0]
    prob  = float(model.predict_proba(values_scaled)[0][1])
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    return label, prob


# =========================================================
# PAGE ROUTES
# =========================================================
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route('/api/user-count')
def user_count():
    count = User.query.count()  # adjust to your model name
    return jsonify({'count': count})

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

    user = db.session.get(User, session["user_id"])

    p = Prediction.query.filter_by(
        user_id=user.id
    ).order_by(Prediction.created_at.desc()).first()

    if p is None:
        class DummyPrediction:
            glucose     = 0
            heart_rate  = 0
            activity    = 0
            probability = 0.0
        p = DummyPrediction()

    return render_template(
        "dashboard.html",
        user=user,
        p=p,
        report_id=int(datetime.now().timestamp()),
        now=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )


# =========================================================
# AUTH ROUTES
# =========================================================
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json

    if Auth.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already exists"}), 409

    auth_record = Auth(email=data["email"], password=data["password"])
    db.session.add(auth_record)
    db.session.commit()

    user = User.query.filter_by(email=data["email"]).first()
    if user:
        session["user_id"] = user.id

    return jsonify({"message": "Signup successful"})


@app.route("/register", methods=["POST"])
def register():
    global active_iot_user_id   # ✅ FIX: update tracker on new registration

    data = request.json

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already registered"}), 409

    user = User(
        email  = data["email"],
        name   = data["name"],
        age    = data["age"],
        gender = data.get("gender"),
        height = data["height"],
        weight = data["weight"],
        bmi    = data["bmi"]
    )
    db.session.add(user)
    db.session.commit()

    session["user_id"]  = user.id
    active_iot_user_id  = user.id   # ✅ ESP32 now writes to this new user

    print(f"[IoT] Active user updated → user_id={active_iot_user_id} ({user.name})")
    return jsonify({"message": "User registered successfully"})


@app.route("/login", methods=["POST"])
def login():
    global active_iot_user_id   # ✅ FIX: update tracker on every login

    data = request.json

    auth_record = Auth.query.filter_by(
        email    = data["email"],
        password = data["password"]
    ).first()

    if not auth_record:
        return jsonify({"error": "Invalid credentials"}), 401

    user = User.query.filter_by(email=auth_record.email).first()
    if not user:
        return jsonify({"error": "User profile not found"}), 404

    session["user_id"] = user.id
    active_iot_user_id = user.id   # ✅ ESP32 now writes to this logged-in user

    print(f"[IoT] Active user updated → user_id={active_iot_user_id} ({user.name})")
    return jsonify({"message": "Login success"})


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/auth")


# =========================================================
# USER INFO
# =========================================================
@app.route("/user")
def user_info():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = db.session.get(User, session["user_id"])
    return jsonify({"name": user.name, "age": user.age})


# =========================================================
# PREDICT (manual input from dashboard form)
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = db.session.get(User, session["user_id"])
    data = request.json or {}

    try:
        glucose    = float(data.get("Glucose"))
        heart_rate = float(data.get("HeartRate"))
        steps      = int(data.get("Activity"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing input data"}), 400

    activity = map_steps_to_activity(steps)
    label, prob = run_prediction(glucose, user.age, user.bmi, heart_rate, activity)

    record = Prediction(
        user_id    = user.id,
        glucose    = glucose,
        heart_rate = heart_rate,
        steps      = steps,
        activity   = activity,
        prediction = label,
        probability= prob
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({"prediction": label, "probability": round(prob, 3)})


# =========================================================
# IOT ROUTES — ESP32 endpoints
# =========================================================
@app.route("/iot/predict", methods=["POST"])
def iot_predict():
    """
    Called by ESP32 every 3 seconds.

    ✅ FIX: uses active_iot_user_id (updated on login/register)
    instead of hardcoded user_id=1.

    Flow:
      User logs in → active_iot_user_id = their user_id
      ESP32 posts here → saved under their user_id
      Their dashboard polls /live → gets their own data ✓
    """
    global active_iot_user_id

    data = request.json or {}

    try:
        glucose    = float(data.get("Glucose"))
        heart_rate = float(data.get("HeartRate"))
        spo2       = float(data.get("SpO2"))   # ✅ read only
        steps      = int(data.get("Activity"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    activity = map_steps_to_activity(steps)

    # Get active user's profile for age & BMI
    user = db.session.get(User, active_iot_user_id)
    if user:
        age = user.age
        bmi = user.bmi
    else:
        # Fallback if no user has logged in yet
        age = 30
        bmi = 23.5

    label, prob = run_prediction(glucose, age, bmi, heart_rate, activity)

    record = Prediction(
        user_id     = active_iot_user_id,   # ✅ dynamic, not hardcoded 1
        glucose     = glucose,
        heart_rate  = heart_rate,
        spo2        = spo2,
        steps       = steps,
        activity    = activity,
        prediction  = label,
        probability = prob
    )
    db.session.add(record)
    db.session.commit()

    print(f"[IoT] Saved → user_id={active_iot_user_id} | glucose={glucose} | HR={heart_rate} | {label} ({prob:.1%})")

    return jsonify({"prediction": label, "probability": round(prob, 3)})


@app.route("/iot/live")
def iot_live():
    """Latest IoT reading for the currently active user (debug/admin view)."""
    record = Prediction.query.filter_by(
        user_id=active_iot_user_id
    ).order_by(Prediction.created_at.desc()).first()

    if not record:
        return jsonify({"error": "No IoT data yet"})

    return jsonify({
        "active_user_id": active_iot_user_id,
        "glucose":        record.glucose,
        "heart_rate":     record.heart_rate,
        "spo2":           record.spo2,
        "steps":          record.steps,
        "activity_level": record.activity,
        "prediction":     record.prediction,
        "probability":    record.probability,
        "time":           record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/iot/history")
def iot_history():
    records = Prediction.query.filter_by(
        user_id=active_iot_user_id
    ).order_by(Prediction.created_at.asc()).all()

    return jsonify([{
        "glucose":     r.glucose,
        "heart_rate":  r.heart_rate,
        "steps":       r.steps,
        "probability": r.probability,
        "time":        r.created_at.strftime("%Y-%m-%d %H:%M:%S")
    } for r in records])

 
# =========================================================
# LIVE DATA (user-specific — dashboard polls this)
# =========================================================
@app.route("/live")
def live():
    """
    Called by dashboard every 3 seconds via fetch("/live").
    Returns the latest prediction for the logged-in user.
    This was already correct — it uses session user_id.
    Now it works because /iot/predict saves to the right user_id.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    record = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.desc()).first()

    if not record:
        return jsonify({"error": "No data"})

    return jsonify({
        "glucose":        record.glucose,
        "heart_rate":     record.heart_rate,
        "spo2":           record.spo2,
        "steps":          record.steps,
        "activity_level": record.activity,
        "probability":    record.probability,
        "prediction":     record.prediction
    })


# =========================================================
# HISTORY
# =========================================================
@app.route("/history")
def history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    records = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.asc()).all()

    return jsonify([{
        "glucose":     r.glucose,
        "heart_rate":  r.heart_rate,
        "steps":       r.steps,
        "activity":    r.activity,
        "probability": r.probability,
        "time":        r.created_at.strftime("%Y-%m-%d %H:%M:%S")
    } for r in records])


# =========================================================
# REPORT
# =========================================================
@app.route("/report")
def report():
    if "user_id" not in session:
        return redirect("/auth")

    user = db.session.get(User, session["user_id"])
    p    = Prediction.query.filter_by(
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


# =========================================================
# STATS
# =========================================================
@app.route("/stats/active-users")
def active_users():
    count = db.session.query(User.id).count()
    return jsonify({"active_users": count})


# =========================================================
# DEBUG — see which user ESP32 is currently writing to
# =========================================================
@app.route("/iot/status")
def iot_status():
    user = db.session.get(User, active_iot_user_id)
    return jsonify({
        "active_iot_user_id": active_iot_user_id,
        "active_user_name":   user.name if user else "Unknown",
        "active_user_email":  user.email if user else "Unknown"
    })


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
