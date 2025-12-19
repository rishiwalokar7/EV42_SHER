import joblib
import numpy as np

try:
    model = joblib.load("model_Malaria_Pneumonia.sav")
    print("Model loaded.")
    
    # Random dummy input for 12 features
    # 'high_fever' 'chills' 'vomiting' 'headache' 'sweating' 'muscle_pain'
    # 'cough' 'phlegm' 'breathlessness' 'chest_pain' 'fast_heart_rate' 'fatigue'
    input_data = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] 
    
    prediction = model.predict(input_data)[0]
    print(f"Prediction: {prediction}")
    print(f"Prediction type: {type(prediction)}")
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        print(f"Probs: {probs}")
        
        # Simulating the error line
        try:
            prob_disease = probs[prediction]
            print(f"Prob at index {prediction}: {prob_disease}")
        except Exception as e:
            print(f"CAUGHT EXPECTED ERROR: {e}")

except Exception as e:
    print(f"General Error: {e}")
