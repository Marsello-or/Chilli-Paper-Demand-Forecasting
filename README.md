# 🌶️ Smart Agro-Supply Chain: Chili Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

> **End-to-End Machine Learning Project** untuk memprediksi permintaan Cabai Rawit guna meminimalisir *Food Waste* dan mengoptimalkan profitabilitas rantai pasok agroindustri.

---

## 🚀 Live Demo
Coba aplikasi dashboard interaktif di sini:  
### [👉 Klik Disini untuk Membuka Aplikasi](https://chilli-pepper-demand-forecasting.streamlit.app/)

---

## 💼 Latar Belakang & Masalah Bisnis

Dalam industri ritel pertanian, **Cabai Rawit** adalah komoditas dengan volatilitas tinggi (*High Volatility*). Tantangan utama yang dihadapi distributor adalah:

1.  **Sifat Mudah Rusak (*Perishability*):** Stok berlebih akan busuk dalam 3-5 hari, menyebabkan kerugian langsung (*Food Waste*).
2.  **Permintaan Inelastis & Musiman:** Saat Hari Raya (Lebaran/Nataru), harga melonjak drastis namun permintaan justru meningkat, seringkali menyebabkan *Stockout* (kehilangan potensi penjualan).
3.  **Inefisiensi Manual:** Perencanaan stok yang hanya mengandalkan "firasat" seringkali meleset hingga 30-40%.

**Solusi Project:** Membangun model Machine Learning untuk memprediksi volume penjualan harian dengan memperhitungkan faktor eksternal (Musim Liburan, Harga, Hari), sehingga stok gudang bisa diatur secara presisi.

---

## 🛠️ Metodologi & Tech Stack

Project ini mencakup siklus lengkap Data Science:

* **Data Processing:** Pembersihan data "kotor" (Dirty Data) simulasi ritel (handling outliers, missing values, string parsing).
* **Feature Engineering:** Membuat fitur `Is_Holiday_Season` menggunakan logika kalender untuk menangkap pola lonjakan permintaan saat Lebaran.
* **Modeling:** Membandingkan performa **XGBoost** (Traditional ML) vs **LSTM** (Deep Learning).
* **Deployment:** Membangun Web App interaktif menggunakan **Streamlit**.

**Teknologi:** `Python`, `Pandas`, `NumPy`, `Scikit-Learn`, `XGBoost`, `TensorFlow/Keras`, `Streamlit`.

---

## 📊 Hasil Evaluasi Model

Kami membandingkan dua algoritma untuk mencari keseimbangan antara akurasi dan efisiensi deployment.

| Model | MAE (Mean Absolute Error) | Error Rate | Keunggulan |
| :--- | :--- | :--- | :--- |
| **LSTM (Deep Learning)** | **15.9 Kg** | **15.9%** | Menangkap tren *sequence* jangka pendek lebih baik. |
| **XGBoost (Selected)** | 16.3 Kg | 16.3% | Komputasi sangat ringan, cepat, dan *explainable*. |

> **Keputusan Deployment:** Meskipun LSTM unggul tipis (0.4%), **XGBoost** dipilih untuk deployment produksi karena efisiensi sumber daya (ringan) dan kecepatan inferensi yang real-time.

---

## 💰 Business Impact (Dampak Bisnis)

Berdasarkan simulasi pada distributor skala menengah (omzet harian ~100 Kg):

* **Pengurangan Error:** Menurunkan tingkat kesalahan prediksi dari ~30% (manual) menjadi ~16% (AI).
* **Potensi Penghematan:** Menyelamatkan sekitar **14 Kg cabai per hari** dari risiko pembusukan.
* **Valuasi Ekonomi:** Estimasi penghematan sebesar **Rp 16.800.000 per bulan** (Asumsi harga rata-rata Rp 40.000/Kg).

---

## 📸 Screenshots

![Dashboard Preview](app-ss.png)

---

## 💻 Cara Menjalankan di Lokal

Jika Anda ingin menjalankan project ini di komputer Anda:

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/Marsello-or/Chilli-Paper-Demand-Forecasting.git](https://github.com/Marsello-or/Chilli-Paper-Demand-Forecasting.git)
    cd nama-repo
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```

---

## 👤 Author

**[Marsello Ormanda]** *Agroindustry Student & Aspiring Data Scientist*

Tertarik mendiskusikan project ini atau kolaborasi?  
Connect with me on [LinkedIn](https://www.linkedin.com/in/marsello-ormanda/) or check my [Portfolio](https://drive.google.com/file/d/14ZcWJSMYheUqy-Z6Qpajqc7vatfexp21/view?usp=sharing).


---


