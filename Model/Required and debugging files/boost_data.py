import pandas as pd
import numpy as np

# 1. Load your original data
input_file = "datasets/SHER_Malaria_Pneumonia.csv"
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"❌ Error: Could not find {input_file}")
    exit()

target_col = "prognosis"

# 2. Filter for ONLY Malaria and Pneumonia
# We search for these words in the diagnosis column
df_filtered = df[df[target_col].astype(str).str.contains("Malaria|Pneumonia", case=False, na=False)]

print(f"Original Count (Filtered): {len(df_filtered)}")

if len(df_filtered) == 0:
    print("❌ Error: No Malaria or Pneumonia rows found to boost!")
    print("Check your CSV to ensure the 'prognosis' column actually contains 'Malaria' or 'Pneumonia'.")
    exit()

# 3. Define a function to generate fake data (Smart Mode)
def generate_synthetic_data(existing_df, target_disease, num_samples=500):
    # Get only the rows for this specific disease
    subset = existing_df[existing_df[target_col].astype(str).str.contains(target_disease, case=False)]
    
    if len(subset) == 0:
        print(f"⚠️ Warning: No existing data found for {target_disease}. Skipping generation.")
        return pd.DataFrame()
    
    # We only create fake data for Numeric columns (Age, BP, Symptoms)
    numeric_cols = subset.select_dtypes(include=[np.number]).columns
    new_data = pd.DataFrame()
    
    for col in numeric_cols:
        # CHECK: Is this a continuous value (like Age) or a binary symptom (0/1)?
        max_val = subset[col].max()
        
        if max_val > 1.5:  
            # --- CONTINUOUS VARIABLE (Age, BP, Glucose) ---
            # We generate random numbers based on the Mean and Standard Deviation
            mean = subset[col].mean()
            std = subset[col].std()
            if np.isnan(std) or std == 0: std = 1  # prevent crash if all values are identical
            
            # Generate random values
            values = np.random.normal(mean, std, num_samples)
            new_data[col] = np.abs(values) # Ensure no negative ages
            
        else:
            # --- BINARY SYMPTOM (0 or 1) ---
            # We treat the mean as the "probability of having the symptom"
            prob = subset[col].mean()
            
            # Safety Clamp: Ensure probability is strictly between 0.0 and 1.0
            prob = max(0.0, min(1.0, prob))
            
            new_data[col] = np.random.choice([0, 1], size=num_samples, p=[1-prob, prob])
        
    # Add the target label back
    new_data[target_col] = target_disease
    return new_data

# 4. Generate 500 Malaria and 500 Pneumonia cases
print("🚀 Generating synthetic patients...")
synthetic_malaria = generate_synthetic_data(df_filtered, "Malaria", 500)
synthetic_pneumonia = generate_synthetic_data(df_filtered, "Pneumonia", 500)

# 5. Combine and Save
final_df = pd.concat([synthetic_malaria, synthetic_pneumonia])

output_filename = "datasets/SHER_Malaria_Pneumonia_Boosted.csv"
final_df.to_csv(output_filename, index=False)

print(f"✅ SUCCESS! Created {len(final_df)} synthetic rows.")
print(f"📂 Saved to: {output_filename}")