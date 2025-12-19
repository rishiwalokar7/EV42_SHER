import joblib
import numpy as np

def get_risk_level(probability):
    if probability < 0.4:
        return "Low"
    elif 0.4 <= probability <= 0.7:
        return "Moderate"
    else:
        return "High"

try:
    model = joblib.load("model_Kidney.sav")
    print("Kidney Model loaded for verification.")
    
    input_data = np.zeros((1, model.n_features_in_))
    
    prediction = model.predict(input_data)[0]
    print(f"Prediction: {prediction}")
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        print(f"Probs: {probs}")
        
        # Emulating the new logic in app.py
        prob_disease = 0.0
        if probs is not None:
            if len(probs) > 1:
                prob_disease = probs[1]
                print("Multiclass/Binary prob used.")
            elif len(probs) == 1:
                print("Single class prob detected.")
                if prediction == 1:
                    prob_disease = probs[0]
                else:
                    prob_disease = 0.0
        else:
            prob_disease = 1.0 if prediction == 1 else 0.0
            
        print(f"Calculated prob_disease: {prob_disease}")
        risk_level = get_risk_level(prob_disease)
        print(f"Risk Level: {risk_level}")
        print("VERIFICATION SUCCESSFUL: No crash.")

except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
