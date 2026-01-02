import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Chili Demand Forecast",
    page_icon="🌶️",
    layout="centered"
)

# --- JUDUL & HEADER ---
st.title("🌶️ AI Chili Demand Forecaster")
st.markdown("Prediksi permintaan Cabai Rawit menggunakan **Machine Learning (XGBoost)** untuk optimasi rantai pasok dan pengurangan *food waste*.")

# --- 1. LOAD MODEL (Menggunakan Joblib .pkl) ---
@st.cache_resource
def load_model():
    try:
        # Pastikan nama file sesuai dengan yang ada di GitHub (.pkl)
        model = joblib.load("model_cabai_xgb.pkl") 
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

model = load_model()

# --- 2. INPUT USER (SIDEBAR) ---
st.sidebar.header("🎛️ Input Parameter")

# A. Input Harga
harga = st.sidebar.number_input("Rencana Harga Jual (Rp/Kg)", min_value=10000, max_value=150000, value=45000, step=1000)

# B. Input Hari
hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
hari_pilihan = st.sidebar.selectbox("Pilih Hari", hari_list)
# Konversi Hari ke Angka (0=Senin, ... 6=Minggu)
day_map = {hari: i for i, hari in enumerate(hari_list)}
day_of_week = day_map[hari_pilihan]

# C. Input Musim
season_options = ["Normal", "Lebaran", "Nataru", "Idul Adha"]
season_pilihan = st.sidebar.selectbox("Musim / Event", season_options)

# --- NEW: AUTHOR CREDENTIALS (SIDEBAR) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Author")
st.sidebar.info("**Marsello Ormanda**")
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/marsello-ormanda/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" alt="Connect on LinkedIn"/>
    </a>
    """, 
    unsafe_allow_html=True
)

# --- 3. LOGIKA PREDIKSI & DATA PREPARATION ---
if st.button("🚀 Prediksi Permintaan"):
    if model:
        # LOGIKA PERBAIKAN (FIX ERROR FEATURE MISMATCH)
        # Model mengharapkan kolom spesifik berikut (urutan tidak boleh salah):
        # ['Harga_Per_Kg', 'Is_Holiday_Season', 'DayOfWeek', 'Nama_Season_Lebaran', 'Nama_Season_Nataru', 'Nama_Season_Normal']
        
        # 1. Tentukan variabel holiday dasar
        is_holiday = 0 if season_pilihan == "Normal" else 1
        
        # 2. Siapkan Dictionary dengan nilai DEFAULT 0 untuk semua kolom Season
        input_dict = {
            'Harga_Per_Kg': [harga],
            'Is_Holiday_Season': [is_holiday],
            'DayOfWeek': [day_of_week],
            'Nama_Season_Lebaran': [0],
            'Nama_Season_Nataru': [0],
            'Nama_Season_Normal': [0]
        }
        
        # 3. Ubah nilai menjadi 1 sesuai pilihan user
        if season_pilihan == "Lebaran":
            input_dict['Nama_Season_Lebaran'] = [1]
        elif season_pilihan == "Nataru":
            input_dict['Nama_Season_Nataru'] = [1]
        elif season_pilihan == "Normal":
            input_dict['Nama_Season_Normal'] = [1]
        # Catatan: Jika pilih "Idul Adha", semua kolom di atas tetap 0 
        # (Karena Idul Adha adalah reference category saat training).

        # 4. Buat DataFrame
        input_df = pd.DataFrame(input_dict)
        
        try:
            # Lakukan Prediksi
            prediction = model.predict(input_df)[0]
            
            # --- TAMPILAN HASIL ---
            st.success(f"📦 Prediksi Permintaan: **{int(prediction)} Kg**")
            
            # --- INSIGHT BISNIS & SUSTAINABILITY ---
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"💡 **Rekomendasi Stok:**\nSiapkan **{int(prediction * 1.1)} Kg**")
                st.caption("*Termasuk buffer stock 10% untuk keamanan.*")
                
            with col2:
                # Hitung potensi waste yang diselamatkan (Asumsi error manual 30%)
                waste_saved = int(prediction * 0.30)
                money_saved = waste_saved * 30000 # Asumsi HPP 30rb
                
                st.metric("🌱 Waste Dicegah", f"{waste_saved} Kg")
                st.caption(f"Hemat potensi rugi ~Rp {money_saved:,}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.write("Debug Data Input:", input_df)

    else:
        st.warning("⚠️ Model belum dimuat. Pastikan file 'model_cabai_xgb.pkl' sudah ada di GitHub.")

else:
    st.info("Silakan atur parameter di sebelah kiri dan klik tombol Prediksi.")
