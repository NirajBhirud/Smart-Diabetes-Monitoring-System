# =========================================================
# SMART DIABETES MONITORING SYSTEM
# MANUAL MODEL TESTING SCRIPT
# =========================================================

import joblib
import numpy as np

print("="*60)
print("MANUAL DIABETES PREDICTION TEST")
print("="*60)


# ---------------------------------------------------------
# STEP 1 — Load model files
# ---------------------------------------------------------

model = joblib.load("ml_model/diabetes_model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")
features = joblib.load("ml_model/feature_names.pkl")

print("\nModel loaded successfully ✅")
print("Expected Features:", features)


# ---------------------------------------------------------
# STEP 2 — Take manual input
# ---------------------------------------------------------

print("\nEnter patient details:\n")

glucose = float(input("Glucose: "))
age = float(input("Age: "))
bmi = float(input("BMI: "))
heart_rate = float(input("Heart Rate: "))
activity = float(input("Activity (0 low, 1 medium, 2 high): "))


# ---------------------------------------------------------
# STEP 3 — Prepare input (same order as training)
# ---------------------------------------------------------

sample = np.array([[glucose, age, bmi, heart_rate, activity]])

sample_scaled = scaler.transform(sample)


# ---------------------------------------------------------
# STEP 4 — Prediction
# ---------------------------------------------------------

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]


# ---------------------------------------------------------
# STEP 5 — Output
# ---------------------------------------------------------

print("\n" + "="*40)

if prediction == 1:
    print("⚠️ Prediction: DIABETIC")
else:
    print("✅ Prediction: NON-DIABETIC")

print("Risk Probability:", round(probability*100, 2), "%")

print("="*40)
