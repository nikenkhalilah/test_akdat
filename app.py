import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Analisis Faktor-Faktor yang Mempengaruhi Kesejahteraan Mental")

# 1. Input Dataset ke Sistem
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:", data.head())

    # 2. Preprocess Data di Sistem
    st.header("Preprocessing Data")
    
    # Menghapus kolom yang tidak relevan (misalnya 'id', 'Name', dll.)
    data = data.drop(columns=['id', 'Name', 'City', 'Profession', 'Degree'], errors='ignore')

    # Menangani nilai kosong
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Encoding variabel kategorikal
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Normalisasi data (selain kolom target)
    scaler = StandardScaler()
    feature_columns = data.drop("Depression", axis=1).columns
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    st.write("Data setelah preprocessing:", data.head())

    # 3. Analisis Data di Sistem (Klasifikasi)
    st.header("Analisis Klasifikasi")
    target_column = "Depression"  # Target kolom yang akan diprediksi
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat model klasifikasi
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Akurasi Model: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # 4. Visualisasi Data di Sistem
    st.header("Visualisasi Faktor-Faktor yang Mempengaruhi Kesejahteraan Mental")
    
    # Feature Importance
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({"Fitur": X.columns, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Fitur", data=importance_df)
    st.pyplot(plt.gcf())
    plt.clf()

    # Distribusi fitur penting berdasarkan kondisi Depression
    st.subheader("Distribusi Fitur Penting Berdasarkan Target")
    important_features = importance_df["Fitur"].head(5).tolist()  # Top 5 fitur
    for feature in important_features:
        st.write(f"Distribusi {feature} Berdasarkan Depression")
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=target_column, y=feature, data=data)
        st.pyplot(plt.gcf())
        plt.clf()

    # Pairplot untuk fitur penting
    st.subheader("Pairplot Fitur Penting")
    sns.pairplot(data, vars=important_features, hue=target_column)
    st.pyplot(plt.gcf())

