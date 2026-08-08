import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('diabetes_model.h5')
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load Model dan Scaler
@st.cache_resource
def load_resources():
    model = load_model('diabetes_model.h5')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_resources()

# Tampilan UI Streamlit
st.set_page_config(page_title="Sistem Prediksi Diabetes ", layout="centered")
st.title(" Sistem Prediksi Diabetes")
st.write("Masukkan parameter klinis pasien untuk memprediksi risiko diabetes berbasis Artificial Neural Network.")
st.divider()

# Input Parameter Klinis Pasien
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glukosa Darah (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Tekanan Darah (mmHg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Ketebalan Lipatan Kulit (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Serum (mu U/ml)", min_value=0, max_value=900, value=79)
    bmi = st.number_input("Indeks Massa Tubuh (BMI kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=30)

st.divider()

# Proses Prediksi
if st.button("Jalankan Prediksi", type="primary"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction_prob = model.predict(input_scaled)[0][0]
    
    st.subheader("Hasil Analisis:")
    if prediction_prob >= 0.5:
        st.error(f"**Risiko Tinggi Diabetes** (Probabilitas: {prediction_prob * 100:.2f}%)")
    else:
        st.success(f"**Risiko Rendah / Normal** (Probabilitas: {(1 - prediction_prob) * 100:.2f}%)")