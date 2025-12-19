import joblib
import numpy as np

try:
    model = joblib.load("model_Kidney.sav")
    print("Kidney Model loaded.")
    
    # Inputs: age, bp, sg, al, su, rbc, sc, hemo
    # Features seem to be 8 selected ones based on app.py, but let's check input size mismatch if any.
    # The previous code in app.py constructs an array of size `model.n_features_in_`.
    
    n_features = model.n_features_in_
    print(f"Features expected: {n_features}")
    
    input_data = np.zeros((1, n_features))
    
    prediction = model.predict(input_data)[0]
    print(f"Prediction: {prediction}")
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        print(f"Probs shape: {probs.shape}")
        print(f"Probs content: {probs}")
        
        try:
            # Replicating the crash
            print(f"Attempting valid access index 0: {probs[0]}")
            print(f"Attempting valid access index 1: {probs[1]}")
        except Exception as e:
            print(f"CRASH REPRODUCED: {e}")

except Exception as e:
    print(f"General Error: {e}")
