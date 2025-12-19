import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. DATASET CONFIGURATION
# ==========================================
dataset_config = {
    "Diabetes": {
        "path": "datasets/Healthcare-Diabetes.csv",
        "target": "Outcome"
    },
    "Kidney": {
        "path": "datasets/Chronic_Kidney_Dsease_data.csv",
        "target": "Diagnosis"
    },
    "Liver": {
        "path": "datasets/Liver_disease_data.csv",
        "target": "Diagnosis"
    },
    "Heart": {
        "path": "datasets/Cardiovascular_Disease_Dataset.csv",
        "target": "target"
    },
    "Hypertension": {
        "path": "datasets/hypertension_dataset.csv",
        "target": "Hypertension"
    },
    "Malaria_Pneumonia": {
        # IMPORTANT: This points to the new synthetic file you just created
        "path": "datasets/SHER_Malaria_Pneumonia_Boosted.csv", 
        "target": "prognosis"
    }
}

# ==========================================
# 2. TRAINING LOOP
# ==========================================
print("🚀 Starting Training Process...")

for disease, config in dataset_config.items():
    print(f"\n----------------\nProcessing {disease}...")
    
    try:
        # A. LOAD DATA
        try:
            df = pd.read_csv(config['path'])
        except FileNotFoundError:
            print(f"❌ Error: File not found at {config['path']}")
            continue

        # B. CLEAN COLUMN NAMES
        df.columns = df.columns.str.strip()
        target_col = config['target']
        
        # C. DROP USELESS COLUMNS
        # We remove IDs to prevent the model from memorizing row numbers
        drop_keywords = ['id', 'ID', 'unnamed', 'Doctor', 'PatientID', 'risk_score', 'risk_level'] 
        cols_to_drop = [c for c in df.columns if any(k in c for k in drop_keywords)]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # D. SPECIFIC DISEASE FIXES
        
        # --- FIX: KIDNEY ---
        # Converts 'ckd'/'notckd' text to 1/0 numbers
        if disease == "Kidney":
            # Force numeric conversion for everything first
            for col in df.columns:
                if col != target_col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # clean target column
            df[target_col] = df[target_col].astype(str).str.strip().str.lower()
            df[target_col] = df[target_col].map({'ckd': 1, 'notckd': 0})
            
            # If mapping failed, try factorize as backup
            if df[target_col].isnull().all():
                df[target_col] = pd.factorize(df[target_col])[0]

        # --- FIX: LIVER ---
        # Liver dataset often uses 1 and 2. We map them to 0 and 1.
        if disease == "Liver":
            if df[target_col].max() > 1:
                df[target_col] = df[target_col].map({2: 0, 1: 1})

        # --- FIX: MALARIA / PNEUMONIA ---
        # Ensure we are strictly predicting these two, even if using boosted data
        if disease == "Malaria_Pneumonia":
            # Filter rows just in case the file has other stuff
            df = df[df[target_col].astype(str).str.contains("Malaria|Pneumonia", case=False, na=False)]

        # E. SEPARATE FEATURES AND TARGET
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # F. HANDLE MISSING VALUES (Imputation)
        # 1. Fill Numbers with Average
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            X[num_cols] = imputer.fit_transform(X[num_cols])

        # 2. Fill Text with "Missing" & Convert to Numbers
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].fillna("Missing")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # G. REMOVE ROWS WITH MISSING TARGETS
        if y.isnull().any():
            X = X[~y.isnull()]
            y = y[~y.isnull()]

        # H. TRAIN THE MODEL
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(X_train) == 0:
            print(f"❌ Error: Not enough data to train {disease}")
            continue

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # I. SAVE THE MODEL
        accuracy = model.score(X_test, y_test)
        filename = f'model_{disease}.sav'
        joblib.dump(model, filename)
        
        print(f"✅ SUCCESS! {disease} Model Accuracy: {accuracy*100:.2f}%")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in {disease}: {e}")

print("\n🎉 All models processed.")