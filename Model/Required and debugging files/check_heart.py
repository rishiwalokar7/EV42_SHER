import joblib

# Load the Heart model
model = joblib.load("model_Heart.sav")

print(f"Model expects {model.n_features_in_} features.")
print("The exact column names are:")
print(model.feature_names_in_)