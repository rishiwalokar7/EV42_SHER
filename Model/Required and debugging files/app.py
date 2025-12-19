import streamlit as st
import joblib
import numpy as np
import pandas as pd
from care_insights import CARE_INSIGHTS

st.set_page_config(page_title="Multi-Disease Predictor", layout="wide")

# ==========================================
# 1. LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Diabetes": "model_Diabetes.sav",
        "Heart": "model_Heart.sav",
        "Kidney": "model_Kidney.sav",
        "Liver": "model_Liver.sav",
        "Hypertension": "model_Hypertension.sav",
        "Malaria_Pneumonia": "model_Malaria_Pneumonia.sav"
    }
    
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(filename)
        except Exception as e:
            st.error(f"Could not load {name} model. Make sure '{filename}' is in the folder.")
    return models

models = load_models()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_risk_level(probability):
    """
    Determine risk level based on probability of the disease.
    """
    if probability < 0.4:
        return "Low"
    elif 0.4 <= probability <= 0.7:
        return "Moderate"
    else:
        return "High"

def display_insights(disease_name, risk_level):
    """
    Display Dos and Donts based on disease and risk level.
    """
    if disease_name in CARE_INSIGHTS and risk_level in CARE_INSIGHTS[disease_name]:
        insights = CARE_INSIGHTS[disease_name][risk_level]
        
        st.markdown(f"### 📋 Care Insights: {risk_level} Risk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("**✅ Do's**")
            for item in insights["Dos"]:
                st.write(f"- {item}")
        
        with col2:
            st.error("**❌ Don'ts**")
            for item in insights["Donts"]:
                st.write(f"- {item}")
    else:
        st.info("No specific insights available for this result.")

def make_prediction_and_display(model_name, input_data, feature_names=None):
    """
    Central function to handle prediction, risk calculation, and display.
    Feature_names is generic placeholder if needed for dataframe conversion, 
    but models here seem to accept arrays.
    """
    try:
        model = models[model_name]
        
        # Get raw prediction
        prediction = model.predict(input_data)[0]
        
        # Get probability if available
        probability = 0.0
        risk_level = "Low"
        disease_detected = False
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
        else:
            probs = None

        # --- LOGIC PER MODEL ---
        if model_name == "Heart":
            # Heart: 0 = High Risk, 1 = Low Risk
            # Probability of Disease (class 0) = probs[0]
            prob_disease = probs[0] if probs is not None else (1.0 if prediction == 0 else 0.0)
            
            risk_level = get_risk_level(prob_disease)
            
            if prediction == 0:
                st.error(f"⚠️ Prediction: Heart Disease Detected (Confidence: {prob_disease:.2%})")
                disease_detected = True
            else:
                if risk_level == "Moderate":
                    st.warning(f"⚠️ Prediction: Low Risk, but Moderate Probability ({prob_disease:.2%})")
                else:
                    st.success(f"✅ Prediction: Heart is Healthy (Confidence: {1-prob_disease:.2%})")
            
            # Display insights based on calculated risk level
            display_insights("Heart", risk_level)

        elif model_name == "Malaria_Pneumonia":
             # 0=Healthy? No, model returns strings 'Malaria' or 'Pneumonia'
             # classes_ = ['Malaria', 'Pneumonia']
             
             # If the model returns a string (e.g. "Pneumonia"), we need to map it to index
             if isinstance(prediction, str):
                 try:
                     # Get index of the predicted class to find its probability
                     class_idx = list(model.classes_).index(prediction)
                     prob_disease = probs[class_idx] if probs is not None else 1.0
                 except (ValueError, IndexError):
                     prob_disease = 1.0
                     
                 risk_level = get_risk_level(prob_disease)
                 
                 st.error(f"⚠️ Prediction: {prediction} Detected (Confidence: {prob_disease:.2%})")
                 display_insights("Malaria_Pneumonia", risk_level)
                 
             else:
                 # Legacy integer handling if generic
                 if prediction == 0:
                     prob_healthy = probs[0] if probs is not None else 1.0
                     st.success(f"✅ Prediction: Healthy (Confidence: {prob_healthy:.2%})")
                     display_insights("Malaria_Pneumonia", "Low")
                 else:
                     condition = "Malaria" if prediction == 1 else "Pneumonia"
                     prob_disease = probs[prediction] if probs is not None else 1.0
                     
                     risk_level = get_risk_level(prob_disease)
                     
                     st.error(f"⚠️ Prediction: {condition} Detected (Confidence: {prob_disease:.2%})")
                     display_insights("Malaria_Pneumonia", risk_level)

        else:
            # Standard Binary: 1 = Disease, 0 = Healthy
            # Diabetes, Liver, Kidney, Hypertension
            
            # Robust probability extraction
            prob_disease = 0.0
            if probs is not None:
                if len(probs) > 1:
                    prob_disease = probs[1]
                elif len(probs) == 1:
                    # Model has only 1 class. 
                    # If that class represents disease (e.g. 1), take the prob. 
                    # If it represents healthy (e.g. 0 or -1 typically considered non-1), take 0.
                    # We check the PREDICTION to determine context.
                    # Original logic was: if prediction == 1 -> Disease.
                    if prediction == 1:
                        prob_disease = probs[0]
                    else:
                        prob_disease = 0.0
            else:
                prob_disease = 1.0 if prediction == 1 else 0.0

            risk_level = get_risk_level(prob_disease)
            
            if prediction == 1:
                st.error(f"⚠️ Prediction: High Risk of {model_name} (Confidence: {prob_disease:.2%})")
                display_insights(model_name, risk_level)
            else:
                if risk_level == "Moderate": # Healthy class predicted but probability of disease is in moderate range (e.g. 0.45)
                     st.warning(f"⚠️ Prediction: Low Risk, but elevated probability ({prob_disease:.2%})")
                     display_insights(model_name, risk_level)
                else:
                     st.success(f"✅ Prediction: Low Risk (Healthy) (Confidence: {1-prob_disease:.2%})")
                     display_insights(model_name, "Low")
            
    except ValueError as e:
        st.error(f"⚠️ Input Error: {e}")
    except KeyError:
        st.error(f"⚠️ Model {model_name} not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ==========================================
# 3. PAGE UI
# ==========================================
st.markdown("<h1 style='text-align: center;'>SHER</h1>", unsafe_allow_html=True)
st.title("🏥 AI-Powered Health Risk Predictor")
st.markdown("Select a disease from the sidebar to assess your risk and get personalized care insights.")

# Sidebar
selected_disease = st.sidebar.selectbox("Select Disease Model", list(models.keys()))

# --- DIABETES ---
if selected_disease == "Diabetes":
    st.header("🍬 Diabetes Risk Assessment")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level", 0, 300, 100)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin Level", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 0, 120, 30)

    if st.button("Predict Diabetes Risk"):
        input_data = [[0, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]] # Note: first 0 might be a dummy column? Keeping original structure.
        # Original: [0, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        # Wait, standard Pima Indians dataset usually has 8 cols. 
        # The previous code passed 9 items: [0, ...]. 
        # I will respect the legacy code's input shape.
        make_prediction_and_display("Diabetes", input_data)


# --- HEART ---
elif selected_disease == "Heart":
    st.header("❤️ Heart Disease Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 0, 120, 50)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP", 50, 250, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting BS > 120?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 50, 250, 150)
        exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3]) 

    if st.button("Predict Heart Disease"):
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]]
        make_prediction_and_display("Heart", input_data)


# --- LIVER ---
elif selected_disease == "Liver":
    st.header("🍺 Liver Disease Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 0, 100, 40)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        total_bil = st.number_input("Total Bilirubin", 0.0, 50.0, 1.0)
        direct_bil = st.number_input("Direct Bilirubin", 0.0, 30.0, 0.5)
        alkphos = st.number_input("Alkaline Phosphotase", 0, 2000, 200)
    with col2:
        sgpt = st.number_input("Alamine Aminotransferase", 0, 2000, 40)
        sgot = st.number_input("Aspartate Aminotransferase", 0, 2000, 40)
        proteins = st.number_input("Total Proteins", 0.0, 10.0, 6.0)
        albumin = st.number_input("Albumin", 0.0, 6.0, 3.0)
        ag_ratio = st.number_input("A/G Ratio", 0.0, 3.0, 1.0)

    if st.button("Predict Liver Disease"):
        input_data = [[age, gender, total_bil, direct_bil, alkphos, sgpt, sgot, proteins, albumin, ag_ratio]]
        make_prediction_and_display("Liver", input_data)


# --- KIDNEY ---
elif selected_disease == "Kidney":
    st.header("💧 Kidney Disease Assessment")
    st.info("Note: Showing key inputs only. Other values will be assumed normal.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 100, 50)
        bp = st.number_input("Blood Pressure", 0, 200, 80)
        sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020)
        al = st.number_input("Albumin", 0, 5, 0)
    with col2:
        su = st.number_input("Sugar", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", [0, 1], format_func=lambda x: "Normal" if x==1 else "Abnormal")
        sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
        hemo = st.number_input("Hemoglobin", 0.0, 20.0, 15.0)

    if st.button("Predict Kidney Disease"):
        # Original logic constructed a specific feature array
        try:
            model_n_features = models["Kidney"].n_features_in_
            features = np.zeros((1, model_n_features))
            
            features[0, 0] = age
            features[0, 1] = bp
            features[0, 2] = sg
            features[0, 3] = al
            features[0, 4] = su
            features[0, 5] = rbc
            features[0, 9] = sc 
            features[0, 10] = hemo 
            
            make_prediction_and_display("Kidney", features)
                
        except Exception as e:
            st.error(f"Error building input data: {e}")


# --- HYPERTENSION ---
elif selected_disease == "Hypertension":
    st.header("🩸 Hypertension Risk Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 40)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        chol = st.number_input("Cholesterol", 100, 400, 200)
    with col2:
        sys_bp = st.number_input("Systolic BP", 80, 250, 120)
        dia_bp = st.number_input("Diastolic BP", 50, 150, 80)
        smoke = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        glucose = st.number_input("Glucose", 50, 300, 100)

    if st.button("Predict Hypertension"):
        try:
            model_n_features = models["Hypertension"].n_features_in_
            features = np.zeros((1, model_n_features))
            
            features[0, 0] = age
            features[0, 1] = sex
            features[0, 2] = bmi
            features[0, 3] = chol
            features[0, 4] = sys_bp
            features[0, 5] = dia_bp
            features[0, 6] = smoke
            features[0, 12] = glucose 
            
            make_prediction_and_display("Hypertension", features)

        except Exception as e:
            st.error(f"Error building input data: {e}")


# --- MALARIA / PNEUMONIA ---
elif selected_disease == "Malaria_Pneumonia":
    st.header("🦟 Malaria & Pneumonia Check")
    st.info("Check all symptoms that apply.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_fever = st.selectbox("High Fever", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        chills = st.selectbox("Chills", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        vomiting = st.selectbox("Vomiting", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        headache = st.selectbox("Headache", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        sweating = st.selectbox("Sweating", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        muscle_pain = st.selectbox("Muscle Pain", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        cough = st.selectbox("Cough", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        phlegm = st.selectbox("Phlegm", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with col3:
        breathlessness = st.selectbox("Breathlessness", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        chest_pain = st.selectbox("Chest Pain", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        fast_heart_rate = st.selectbox("Fast Heart Rate", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    if st.button("Predict Condition"):
        input_data = [[
            high_fever, chills, vomiting, headache, 
            sweating, muscle_pain, cough, phlegm, 
            breathlessness, chest_pain, fast_heart_rate, fatigue
        ]]
        
        make_prediction_and_display("Malaria_Pneumonia", input_data)

st.markdown("---")
st.caption("Disclaimer: This AI tool is for educational purposes only.")