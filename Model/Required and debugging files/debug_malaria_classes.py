import joblib

try:
    model = joblib.load("model_Malaria_Pneumonia.sav")
    print(f"Classes: {model.classes_}")
    
except Exception as e:
    print(f"Error: {e}")
