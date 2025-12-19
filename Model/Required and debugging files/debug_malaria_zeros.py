import joblib
import numpy as np

try:
    model = joblib.load("model_Malaria_Pneumonia.sav")
    
    # All zeros
    input_data = [[0] * 12]
    
    prediction = model.predict(input_data)[0]
    print(f"Prediction (zeros): {prediction}")
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        print(f"Probs (zeros): {probs}")

except Exception as e:
    print(f"Error: {e}")
