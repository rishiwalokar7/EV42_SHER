import joblib

# Load your model
model = joblib.load('model_Malaria_Pneumonia.sav')

print(f"Expecting: {model.n_features_in_} features")

# Try to print the names of the columns
if hasattr(model, 'feature_names_in_'):
    print("Column Names:", model.feature_names_in_)
else:
    print("The model didn't save column names. You need to check your training dataset (csv).")