import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Konfigurasi tampilan awal
st.set_page_config(page_title="Analisis Klaster COVID-19", layout="wide")
st.title("📊 Analisis Klaster Harian COVID-19 di Indonesia")

# Upload file
uploaded_file = st.file_uploader("📎 Upload file CSV dengan data COVID-19", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧾 Pratinjau Data")
    st.dataframe(df.head())

    # Pilih kolom numerik
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.subheader("⚙️ Konfigurasi Klastering")
    selected_cols = st.multiselect(
        "Pilih Kolom untuk Clustering (min 2 kolom):",
        numeric_cols,
        default=['jumlah_positif', 'jumlah_dirawat'] if 'jumlah_positif' in numeric_cols else numeric_cols[:2]
    )

    n_clusters = st.slider("Jumlah Klaster (k)", min_value=2, max_value=10, value=3)

    if len(selected_cols) >= 2:
        # Standarisasi data
        data_scaled = StandardScaler().fit_transform(df[selected_cols].dropna())
        clustering_data = df[selected_cols].dropna().copy()

        # Jalankan KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_scaled)

        # Gabungkan kembali ke df asli
        df.loc[clustering_data.index, 'Cluster'] = labels
        df['Cluster'] = df['Cluster'].astype(int)

        # Visualisasi scatter plot klaster
        st.subheader("📍 Visualisasi Klaster (2D Scatterplot)")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x=selected_cols[0], y=selected_cols[1], hue='Cluster', palette='Set1', ax=ax1)
        ax1.set_xlabel(selected_cols[0])
        ax1.set_ylabel(selected_cols[1])
        ax1.set_title("Visualisasi Klaster")
        st.pyplot(fig1)

        # Statistik ringkasan
        st.subheader("📊 Statistik Rata-rata per Klaster")
        st.dataframe(df.groupby('Cluster')[selected_cols].mean().round(2))

        # Visualisasi tren waktu (jika ada kolom tanggal)
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                st.subheader("📈 Tren Waktu Kasus Positif per Klaster")

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                for cluster in sorted(df['Cluster'].unique()):
                    df_cluster = df[df['Cluster'] == cluster].sort_values(by='date')
                    ax2.plot(df_cluster['date'], df_cluster[selected_cols[0]], label=f'Klaster {cluster}')

                ax2.set_xlabel("Tanggal")
                ax2.set_ylabel(selected_cols[0])
                ax2.set_title(f"Tren Harian {selected_cols[0]} per Klaster")
                ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=7))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend(title="Klaster")
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Kolom 'date' tidak bisa diubah ke format tanggal: {e}")
    else:
        st.warning("📌 Pilih setidaknya 2 kolom numerik untuk melakukan clustering.")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
