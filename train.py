import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# 1. Load Dataset Pima Indians Diabetes
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. Pemisahan Fitur (X) dan Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Split Data Train & Test (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalisasi Data (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan Scaler untuk Streamlit
joblib.dump(scaler, 'scaler.joblib')

# 5. Arsitektur Model ANN (Backpropagation)
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 6. Kompilasi & Training Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Memulai proses training model ANN...")
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=1)

# 7. Evaluasi & Simpan Model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%")
model.save('diabetes_model.h5')
print("Model berhasil disimpan sebagai 'diabetes_model.h5' dan 'scaler.joblib'")