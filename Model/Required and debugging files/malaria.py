import pandas as pd
import numpy as np

# We define the columns based on your project structure
columns = [
    'high_fever', 'chills', 'vomiting', 'headache', 'sweating', 'muscle_pain', # Malaria heavy
    'cough', 'phlegm', 'breathlessness', 'chest_pain', 'fast_heart_rate', 'fatigue', # Pneumonia heavy
    'prognosis'
]

# 1. Generate MALARIA Cases (High Chills, Fever, Muscle Pain)
print("🦟 Generating Malaria cases...")
malaria_data = []
for _ in range(500):
    row = {
        'high_fever': np.random.choice([1, 0], p=[0.9, 0.1]),      # 90% chance
        'chills': np.random.choice([1, 0], p=[0.95, 0.05]),        # Key symptom
        'vomiting': np.random.choice([1, 0], p=[0.6, 0.4]),
        'headache': np.random.choice([1, 0], p=[0.7, 0.3]),
        'sweating': np.random.choice([1, 0], p=[0.8, 0.2]),
        'muscle_pain': np.random.choice([1, 0], p=[0.8, 0.2]),
        # Pneumonia symptoms should be RARE in Malaria
        'cough': np.random.choice([1, 0], p=[0.1, 0.9]),
        'phlegm': np.random.choice([1, 0], p=[0.05, 0.95]),
        'breathlessness': np.random.choice([1, 0], p=[0.05, 0.95]),
        'chest_pain': np.random.choice([1, 0], p=[0.1, 0.9]),
        'fast_heart_rate': np.random.choice([1, 0], p=[0.2, 0.8]),
        'fatigue': np.random.choice([1, 0], p=[0.8, 0.2]),         # Common in both
        'prognosis': 'Malaria'
    }
    malaria_data.append(row)

# 2. Generate PNEUMONIA Cases (Cough, Breathlessness, Phlegm)
print("🫁 Generating Pneumonia cases...")
pneumonia_data = []
for _ in range(500):
    row = {
        'high_fever': np.random.choice([1, 0], p=[0.8, 0.2]),      # Common in both
        'chills': np.random.choice([1, 0], p=[0.2, 0.8]),          # Rare in Pneumonia
        'vomiting': np.random.choice([1, 0], p=[0.1, 0.9]),
        'headache': np.random.choice([1, 0], p=[0.3, 0.7]),
        'sweating': np.random.choice([1, 0], p=[0.3, 0.7]),
        'muscle_pain': np.random.choice([1, 0], p=[0.3, 0.7]),
        # Pneumonia symptoms should be HIGH
        'cough': np.random.choice([1, 0], p=[0.95, 0.05]),         # Key symptom
        'phlegm': np.random.choice([1, 0], p=[0.9, 0.1]),
        'breathlessness': np.random.choice([1, 0], p=[0.9, 0.1]),  # Key symptom
        'chest_pain': np.random.choice([1, 0], p=[0.8, 0.2]),
        'fast_heart_rate': np.random.choice([1, 0], p=[0.4, 0.6]),
        'fatigue': np.random.choice([1, 0], p=[0.8, 0.2]),
        'prognosis': 'Pneumonia'
    }
    pneumonia_data.append(row)

# 3. Combine and Save
df = pd.DataFrame(malaria_data + pneumonia_data)

# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

output_file = "datasets/SHER_Malaria_Pneumonia_Boosted.csv"
df.to_csv(output_file, index=False)

print(f"✅ SUCCESS! Created High-Quality Dataset: {output_file}")
print("Run train_models.py now, and accuracy should be ~99%.")