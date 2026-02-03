# =========================================================
# SMART DIABETES MONITORING SYSTEM
# FINAL PRODUCTION RANDOM FOREST MODEL
# SIMPLE • FAST • CLOUD FRIENDLY
# =========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


print("="*60)
print("FINAL WEARABLE DIABETES MODEL TRAINING")
print("="*60)


# ---------------------------------------------------------
# STEP 1 — Load Dataset
# ---------------------------------------------------------

data = pd.read_csv("dataset/diabetes.csv")

print("\nDataset Loaded:", data.shape)
print("Class Distribution:\n", data["Outcome"].value_counts())


# ---------------------------------------------------------
# STEP 2 — Basic Cleaning
# Replace impossible zeros
# ---------------------------------------------------------

for col in ["Glucose", "BMI"]:
    median = data[col].median()
    data[col] = data[col].replace(0, median)

print("\nData cleaned")


# ---------------------------------------------------------
# STEP 3 — Add Wearable Sensor Features (simulation)
# ---------------------------------------------------------

np.random.seed(42)

# Heart Rate (realistic values)
data["HeartRate"] = (
    70
    + (data["Glucose"] - 110) / 10
    + (data["BMI"] - 25) / 3
    + np.random.normal(0, 4, len(data))
).clip(50, 120)

# Activity level (0 low, 1 medium, 2 high)
data["Activity"] = np.random.randint(0, 3, len(data))


# ---------------------------------------------------------
# STEP 4 — Final Features (ONLY wearable inputs)
# ---------------------------------------------------------

features = ["Glucose", "Age", "BMI", "HeartRate", "Activity"]

X = data[features]
y = data["Outcome"]

print("\nUsing Features:", features)


# ---------------------------------------------------------
# STEP 5 — Train Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ---------------------------------------------------------
# STEP 6 — Scaling (robust for outliers)
# ---------------------------------------------------------

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------
# STEP 7 — Train Random Forest (simple & powerful)
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",   # handles imbalance automatically
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel trained")


# ---------------------------------------------------------
# STEP 8 — Evaluation
# ---------------------------------------------------------

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n" + "="*40)
print("MODEL PERFORMANCE")
print("="*40)

print("Accuracy:", round(acc*100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))


# ---------------------------------------------------------
# STEP 9 — Cross Validation
# ---------------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv)

print("\nCross Validation Accuracy:", round(scores.mean()*100, 2), "%")


# ---------------------------------------------------------
# STEP 10 — Save Files (for Flask/cloud)
# ---------------------------------------------------------

joblib.dump(model, "ml_model/diabetes_model.pkl")
joblib.dump(scaler, "ml_model/scaler.pkl")
joblib.dump(features, "ml_model/feature_names.pkl")

print("\n✅ Saved:")
print("  diabetes_model.pkl")
print("  scaler.pkl")
print("  feature_names.pkl")

print("\n🎉 TRAINING COMPLETE")
print("="*60)
