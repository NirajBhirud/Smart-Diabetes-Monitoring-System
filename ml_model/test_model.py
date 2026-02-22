# =========================================================
# SMART DIABETES MONITORING SYSTEM
# MANUAL MODEL TESTING (CLINICALLY CORRECT)
# =========================================================

import joblib
import numpy as np
import pandas as pd

print("="*60)
print("MANUAL DIABETES PREDICTION TEST")
print("="*60)

# ---------------- LOAD FILES ----------------
model    = joblib.load("ml_model/diabetes_model.pkl")
scaler   = joblib.load("ml_model/scaler.pkl")
features = joblib.load("ml_model/feature_names.pkl")

# ---- Clinical constants (MUST MATCH TRAINING) ----
BASE_WEIGHTS = np.array([5.0, 1.2, 1.5, 0.9, 0.7, 4.0, 6.0, 8.0])
ZONE_MULT    = {0: 1.0, 1: 1.5, 2: 3.0}
GLUCOSE_IDX  = features.index("Glucose")
ZONE_IDX     = features.index("Glucose_Zone")

def engineer_glucose_features(glucose):
    if glucose <= 99:
        zone = 0
    elif glucose <= 125:
        zone = 1
    else:
        zone = 2
    excess  = max(0, glucose - 125)
    is_diab = 1 if glucose >= 126 else 0
    return zone, excess, is_diab

def apply_weights(X):
    Xw = X.astype(float) * BASE_WEIGHTS
    zones = X[:, ZONE_IDX].astype(int)
    Xw[:, GLUCOSE_IDX] *= np.array([ZONE_MULT[z] for z in zones])
    return Xw

# ---------------- INPUT ----------------
print("\nEnter patient details:\n")

glucose     = float(input("Glucose: "))
age         = float(input("Age: "))
bmi         = float(input("BMI: "))
heart_rate  = float(input("Heart Rate: "))
activity    = int(input("Activity (0 low, 1 medium, 2 high): "))

zone, excess, is_diab = engineer_glucose_features(glucose)

row = pd.DataFrame(
    [[glucose, age, bmi, heart_rate, activity, zone, excess, is_diab]],
    columns=features
)

# ---------------- SCALE + PREDICT ----------------
X_scaled = scaler.transform(apply_weights(row.values))

pred = model.predict(X_scaled)[0]
prob = model.predict_proba(X_scaled)[0][1]
print("TEST scaler features:", scaler.n_features_in_)
print("\n" + "="*40)
print("⚠️ Prediction: DIABETIC" if pred else "✅ Prediction: NON-DIABETIC")
print("Risk Probability:", round(prob*100, 2), "%")
print("="*40)