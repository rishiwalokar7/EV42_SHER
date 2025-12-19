import joblib
import os

model_files = {
    "Diabetes": "model_Diabetes.sav",
    "Heart": "model_Heart.sav",
    "Kidney": "model_Kidney.sav",
    "Liver": "model_Liver.sav",
    "Hypertension": "model_Hypertension.sav",
    "Malaria_Pneumonia": "model_Malaria_Pneumonia.sav"
}

print("Checking models for predict_proba attribute...")

for name, filename in model_files.items():
    if os.path.exists(filename):
        try:
            model = joblib.load(filename)
            has_proba = hasattr(model, "predict_proba")
            print(f"{name}: predict_proba = {has_proba}")
        except Exception as e:
            print(f"{name}: Error loading - {e}")
    else:
        print(f"{name}: File not found")
