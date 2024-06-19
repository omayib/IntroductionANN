import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data dalam bentuk tabel
data = {
    'Jam Belajar (Jam)': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Nilai Ujian': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
}

if __name__=="__main__":
    # Membuat DataFrame
    df = pd.DataFrame(data)

    # Menampilkan tabel
    print(df)

    # Plotting menggunakan DataFrame Pandas
    # Memisahkan fitur dan label
    X = df[['Jam Belajar (Jam)']]
    y = df['Nilai Ujian']

    # Membagi data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mencetak hasil split data
    print("Data Latih (X_train):")
    print(X_train)
    print("\nData Uji (X_test):")
    print(X_test)
    print("\nNilai Ujian Latih (y_train):")
    print(y_train)
    print("\nNilai Ujian Uji (y_test):")
    print(y_test)

    # Membuat model regresi linier
    model = LinearRegression()

    # Melatih model
    model.fit(X_train, y_train)

    # Membuat prediksi pada data testing
    y_pred = model.predict(X_test)

    # Menampilkan hasil prediksi
    prediksi_df = pd.DataFrame({'Jam Belajar (Jam)': X_test['Jam Belajar (Jam)'], 'Nilai Ujian Aktual': y_test,
                                'Nilai Ujian Prediksi': y_pred})
    print(prediksi_df)

    # Visualisasi hasil prediksi
    plt.figure(figsize=(10, 6))

    # Plot data latih
    plt.scatter(X_train, y_train, color='blue', label='Data Latih', marker='o')

    # Plot data uji
    plt.scatter(X_test, y_test, color='red', label='Data Uji', marker='x')

    # Plot hasil prediksi data uji
    plt.scatter(X_test, y_pred, color='green', label='Prediksi', marker='*')

    # Plot garis regresi
    plt.plot(X, model.predict(X), color='green', linestyle='-', linewidth=2, label='Garis Regresi Linier')

    # Pengaturan plot
    plt.title('Jam Belajar vs Nilai Ujian')
    plt.xlabel('Jam Belajar (Jam)')
    plt.ylabel('Nilai Ujian')
    plt.legend()
    plt.grid(True)

    plt.savefig('plot_jam_belajar_vs_nilai_ujian.png')
    # Menampilkan plot
    plt.show()