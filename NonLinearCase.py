import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data dalam bentuk tabel dengan fitur tambahan
data = {
    'Luas Rumah (sq ft)': [1200, 1500, 800, 600, 2000, 1800],
    'Jumlah Kamar': [3, 4, 2, 1, 5, 4],
    'Lokasi (kode)': [1, 2, 1, 0, 2, 2],
    'Harga Rumah (USD)': [300000, 450000, 200000, 150000, 500000, 480000],
    'Jumlah Kamar Mandi': [2, 3, 1, 1, 4, 3],
    'Tahun Dibangun': [2000, 2010, 1990, 1985, 2015, 2012],
    'Ukuran Lot (sq ft)': [5000, 6000, 4000, 3000, 7000, 6500]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Menampilkan tabel
print(df)

# Memisahkan fitur dan label
X = df[['Luas Rumah (sq ft)', 'Jumlah Kamar', 'Lokasi (kode)', 'Jumlah Kamar Mandi', 'Tahun Dibangun', 'Ukuran Lot (sq ft)']].values
y = df['Harga Rumah (USD)'].values

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"x_train before normalized {X_train}")
# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"x_train after normalized {X_train}")

# Membuat model ANN
model = keras.Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(X, y, epochs=100, verbose=1)
# Membuat prediksi pada data testing
y_pred = model.predict(X_test)
# Menampilkan hasil prediksi
prediksi_df = pd.DataFrame({'Harga Rumah Aktual': y_test, 'Harga Rumah Prediksi': y_pred.flatten()})
print("\nHasil Prediksi:")
print(prediksi_df)
# Prediksi harga rumah baru
X_new = np.array([[1300, 3, 1, 2, 2005, 5500]])
X_new = (X_new - X.mean(axis=0)) / X.std(axis=0)
prediksi_harga = model.predict(X_new)

print("Prediksi harga rumah:", prediksi_harga)

# Mencetak ringkasan model
model.summary()

# Menyimpan diagram arsitektur model ke file gambar
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Plotting Data
plt.figure(figsize=(10, 6))
plt.scatter(df['Luas Rumah (sq ft)'], df['Harga Rumah (USD)'], color='blue', label='Data Rumah')
plt.title('Luas Rumah vs Harga Rumah')
plt.xlabel('Luas Rumah (sq ft)')
plt.ylabel('Harga Rumah (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot untuk Luas Rumah vs Harga Rumah
plt.figure(figsize=(10, 6))
plt.scatter(df['Luas Rumah (sq ft)'], df['Harga Rumah (USD)'], color='blue', label='Luas Rumah vs Harga Rumah')
plt.title('Luas Rumah vs Harga Rumah')
plt.xlabel('Luas Rumah (sq ft)')
plt.ylabel('Harga Rumah (USD)')
plt.legend()
plt.grid(True)
plt.savefig('Luas Rumah vs Harga Rumah')
plt.show()

# Scatter Plot untuk Jumlah Kamar vs Harga Rumah
plt.figure(figsize=(10, 6))
plt.scatter(df['Jumlah Kamar'], df['Harga Rumah (USD)'], color='green', label='Jumlah Kamar vs Harga Rumah')
plt.title('Jumlah Kamar vs Harga Rumah')
plt.xlabel('Jumlah Kamar')
plt.ylabel('Harga Rumah (USD)')
plt.legend()
plt.grid(True)
plt.savefig('Jumlah Kamar vs Harga Rumah')
plt.show()

# Heatmap untuk melihat korelasi antara semua fitur
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Fitur')
plt.savefig('Heatmap Korelasi Fitur')
plt.show()