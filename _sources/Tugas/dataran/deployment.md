---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 6. Deployment (Penyebaran Model via Streamlit)

## 6.1. Arsitektur Aplikasi Web Streamlit
Model dan visualisasi peta spasial diintegrasikan ke dalam antarmuka berbasis web interaktif pada file [`app.py`](file:///app.py):

* **Presentation Layer**: Menampilkan 4 kartu ringkasan metrik utama (Akurasi Model KNN **87.5%** pada data uji piksel & **88.1%** agregat poligon, 635K Piksel Sentinel-2A, 360 Poligon Ground Truth, dan 43 Poligon Salah Klasifikasi).
* **Kontrol Tata Letak**: Pengguna dapat memilih tata letak 3 peta (*Tab Navigasi 1 Peta per Tab*, *Tampilkan Sekaligus Berurutan Vertikal*, atau *Peta 1 & 2 Bersisian 2 Kolom*) serta mengatur ketinggian frame peta dinamis (450px s.d. 900px).
* **Eksplorasi Data Tabular**: Tabel interaktif hasil evaluasi 360 poligon uji dilengkapi filter kategori tutupan lahan, filter status klasifikasi (*Semua Poligon*, *Hanya Prediksi Benar*, atau *Hanya Salah Prediksi*), dan tombol unduh data CSV (`prediksi_poligon_knn_jatim.csv`).
* **Kinerja Cepat & Ringan**: Aplikasi memuat peta HTML yang telah di-generate sebelumnya (*pre-rendered Folium layers*) sehingga tidak membebani memori CPU/RAM server cloud dengan proses pelatihan berulang.

---

## 6.2. Akses Publik & URL Deployment
Aplikasi telah dideploy ke Streamlit Community Cloud dan dapat diakses publik secara luas melalui tautan berikut:

🌐 **[https://uts-psd-lulc.streamlit.app/](https://uts-psd-lulc.streamlit.app/)**

```{raw} html
<div style="margin: 20px 0; border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
    <div style="background: #f6f8fa; padding: 10px 16px; border-bottom: 1px solid #d0d7de; display: flex; justify-content: space-between; align-items: center; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; font-size: 13px;">
        <span>Live Dashboard Web Streamlit (Tutupan Lahan Jawa Timur)</span>
        <div>
            <a href="https://uts-psd-lulc.streamlit.app/" target="_blank" style="color: #6D4C41; text-decoration: none; font-weight: 600; padding: 4px 10px; background: #F8F4EE; border: 1px solid #EADCCF; border-radius: 4px;">↗ Buka di Tab Baru</a>
        </div>
    </div>
    <iframe src="https://uts-psd-lulc.streamlit.app/?embed=true" width="100%" height="750px" style="border: none; display: block;"></iframe>
</div>
```

---

## 6.3. Panduan Menjalankan Aplikasi Secara Lokal
Untuk menjalankan dashboard Streamlit di komputer lokal:

```bash
# 1. Pastikan dependensi telah terinstal
pip install streamlit folium pandas numpy pillow rasterio scikit-learn

# 2. Jalankan aplikasi Streamlit
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di peramban web (*browser*) lokal pada alamat `http://localhost:8501`.
