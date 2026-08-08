import joblib
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load model dan scaler
model = load_model("diabetes_model.h5")
scaler = joblib.load("scaler.joblib")

# 2. Data uji sampel: [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
sample_input = np.array([[1, 120, 70, 20, 79, 25.0, 0.5, 30]])

# 3. Transformasi data dan jalankan prediksi
input_scaled = scaler.transform(sample_input)
prediction_prob = float(model.predict(input_scaled)[0][0])

# 4. Tampilkan hasil di terminal
print("=== HASIL UJI PREDIKSI ===")
print(f"Probabilitas Raw : {prediction_prob:.4f}")
print(f"Persentase Risiko: {prediction_prob * 100:.2f}%")

if prediction_prob >= 0.5:
    print("Status          : Risiko Tinggi Diabetes")
else:
    print(f"Status          : Risiko Rendah / Normal (Keamanan: {(1 - prediction_prob) * 100:.2f}%)")