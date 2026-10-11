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

# 4. Visualisasi Map (Folium)

Di tahap terakhir, kita memvisualisasikan kembali wilayah yang sudah kita digitasi ke dalam sebuah peta *(interactive map)*. 

Mengingat dataset berisi ratusan ribu titik piksel, me-render tiap titik satu-per-satu di peta interaktif akan membekukan (*crash*) memori browser. Solusi paling *robust* adalah menyatukan kembali titik-titik tersebut per `poligon_id` dan membuat jaring area **(Convex Hull)** luarnya, sehingga titik berubah menjadi hamparan wilayah (*Polygon*).

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd
import folium
import geopandas as gpd

# Load Data
df = pd.read_csv('../../Data/UTS/piksel_s2_zaenal.csv')

# 1. Rekonstruksi Titik menjadi Geometri menggunakan GeoPandas
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
)

# Gabungkan (dissolve) berdasarkan poligon_id untuk menyatukan piksel per poligon
polygons = gdf.dissolve(by='poligon_id', aggfunc='first').reset_index()

# Ubah kumpulan titik menjadi poligon tertutup (Convex Hull) dan buang geometri kosong
polygons['geometry'] = polygons['geometry'].convex_hull
polygons = polygons[~polygons['geometry'].is_empty]

# 2. Inisiasi Peta Folium
map_center = [df['lat'].mean(), df['lon'].mean()]
m = folium.Map(
    location=map_center, 
    zoom_start=10, 
    tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', # Standard Google Maps
    attr='Google'
)

# Tambahan Opsi Peta Satelit ke LayerControl
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(m)

# 3. Pewarnaan Poligon Berdasarkan Klasifikasi/Label Asli
colors = {
    'Sawah': '#FFD92F',
    'Bangunan': '#E41A1C',
    'Mangrove': '#8E44AD',
    'Hutan': '#2E7D32',
    'Laut': '#0D47A1',
    'Danau': '#4FC3F7'
}

feature_groups = {}
for label in polygons['label_teks'].unique():
    label_str = str(label)
    feature_groups[label_str] = folium.FeatureGroup(name=label_str)
    m.add_child(feature_groups[label_str])

# Memplot poligon ke peta
for idx, row in polygons.iterrows():
    label_str = str(row['label_teks'])
    color = colors.get(label_str, 'black')
    
    try:
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=f"{label_str} (Poligon ID: {row['poligon_id']})"
        ).add_to(feature_groups[label_str])
    except Exception:
        pass # Lewati apabila terdapat anomali geometri (contoh poligon berupa titik lurus)

folium.LayerControl().add_to(m)
m
```

### Visualisasi Prediksi Model Terbaik (KNN)

Berdasarkan tahap pemodelan sebelumnya, algoritma **K-Nearest Neighbors (KNN)** sering kali memberikan akurasi yang lebih andal untuk klasifikasi data spasial ini. 

Sebagai penutup, mari kita gunakan model KNN (dengan $K=5$) untuk memprediksi kelas dari masing-masing poligon (berdasarkan nilai rata-rata spektral piksel di dalamnya) dan memvisualisasikan hasil prediksinya ke dalam peta interaktif. Hal ini berguna untuk membandingkan apakah prediksi model selaras dengan *ground truth* (label asli).

```{code-cell} ipython3
:tags: [hide-input]

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Menentukan Fitur (X) dan Target (y)
fitur = ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI', 'MNDWI', 'NDBI']
df_bersih = df.dropna(subset=fitur + ['label_teks']).copy()

# Ambil sampel acak berimbang per kelas (600 per kelas)
# Mempertahankan karakteristik spektral asli piksel namun memungkinkan inferensi cepat (~1 menit)
df_sample = df_bersih.groupby('label_teks', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 600), random_state=42)
)

X_train = df_sample[fitur]
y_train = df_sample['label_teks']

# 2. Standarisasi Fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Inisialisasi dan Pelatihan Model KNN (k=5)
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

# 4. Agregasi fitur per poligon (menggunakan rata-rata piksel)
poly_features = df_bersih.groupby('poligon_id')[fitur].mean().reset_index()
poly_features_scaled = scaler.transform(poly_features[fitur])

# Prediksi label poligon menggunakan model KNN
poly_features['prediksi_label'] = knn_model.predict(poly_features_scaled)

# Gabungkan hasil prediksi dengan geometri dan label asli poligon sebelumnya
polygons_pred = polygons[['poligon_id', 'geometry', 'label_teks']].merge(poly_features[['poligon_id', 'prediksi_label']], on='poligon_id')

# 3. Plotting ke Peta Baru (Peta Prediksi)
m_pred = folium.Map(
    location=map_center, 
    zoom_start=10, 
    tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    attr='Google'
)

# Tambahkan opsi satelit
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Esri Satellite', overlay=False, control=True
).add_to(m_pred)

# Siapkan FeatureGroup untuk tiap kelas prediksi dan satu grup khusus "Salah Prediksi"
feature_groups_pred = {}
for label in polygons_pred['prediksi_label'].unique():
    label_str = str(label)
    feature_groups_pred[label_str] = folium.FeatureGroup(name=f"Prediksi: {label_str}")
    m_pred.add_child(feature_groups_pred[label_str])

salah_prediksi_group = folium.FeatureGroup(name="⚠️ Salah Prediksi", show=True)
m_pred.add_child(salah_prediksi_group)

# Memplot poligon dengan warna berdasarkan kelas prediksi
for idx, row in polygons_pred.iterrows():
    label_pred = str(row['prediksi_label'])
    label_asli = str(row['label_teks'])
    color = colors.get(label_pred, 'black')
    is_correct = (label_pred == label_asli)
    centroid = row['geometry'].centroid
    
    try:
        # Plot dasar (sesuai prediksi)
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=f"Prediksi: {label_pred} | Asli: {label_asli} (ID: {row['poligon_id']})"
        ).add_to(feature_groups_pred[label_pred])

        # Plot tambahan (highlight outline merah dan tanda seru) jika salah prediksi
        if not is_correct:
            folium.GeoJson(
                data=row['geometry'].__geo_interface__,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'red',
                    'weight': 4,
                    'dashArray': '5, 5'
                },
                tooltip=f"⚠️ SALAH PREDIKSI! Asli: {label_asli}, Prediksi: {label_pred} (ID: {row['poligon_id']})"
            ).add_to(salah_prediksi_group)
            
            # Tambahkan marker Tanda Seru (!) pada poligon yang salah diprediksi
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.Icon(color='red', icon='exclamation', prefix='fa'),
                tooltip=f"⚠️ SALAH PREDIKSI! Asli: {label_asli} | Prediksi: {label_pred} (ID: {row['poligon_id']})"
            ).add_to(salah_prediksi_group)
    except Exception:
        pass

folium.LayerControl().add_to(m_pred)
m_pred
```
```{code-cell} ipython3
:tags: [hide-input]

import os
import warnings
import rasterio
import numpy as np
import folium
from scipy.ndimage import label, binary_closing, binary_fill_holes, binary_dilation

warnings.filterwarnings('ignore')

# Fungsi pembantu: konversi kode warna hex (#RRGGBB) ke (R, G, B)
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# Tentukan path raster wilayah utuh
FILE_RASTER = '../../Data/UTS/jatim_250m.tif'
if not os.path.exists(FILE_RASTER):
    for p in ['jatim_250m.tif', '../Data/UTS/jatim_250m.tif', '../../Data/UTS/jatim_250m.tif']:
        if os.path.exists(p):
            FILE_RASTER = p
            break

m_full = None

if os.path.exists(FILE_RASTER):
    with rasterio.open(FILE_RASTER) as src:
        b2 = src.read(1).astype(float)
        b3 = src.read(2).astype(float)
        b4 = src.read(3).astype(float)
        b8 = src.read(4).astype(float)
        b11 = src.read(5).astype(float)
        
        bounds = src.bounds
        img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

    e = 1e-9
    ndvi = (b8 - b4) / (b8 + b4 + e)
    ndwi = (b3 - b8) / (b3 + b8 + e)
    mndwi = (b3 - b11) / (b3 + b11 + e)
    ndbi = (b11 - b8) / (b11 + b8 + e)
    
    height, width = b2.shape
    valid_mask = (b2 > 0) & (b2 != -32768) & (b8 > 0) & (b11 > 0)
    valid_mask_1d = valid_mask.flatten()
    
    features_stack = np.column_stack((
        b2.flatten(), b3.flatten(), b4.flatten(), b8.flatten(), b11.flatten(),
        ndvi.flatten(), ndwi.flatten(), mndwi.flatten(), ndbi.flatten()
    ))
    
    pred_1d = np.full(height * width, fill_value="", dtype=object)
    
    if np.any(valid_mask_1d):
        features_valid = features_stack[valid_mask_1d]
        features_scaled = scaler.transform(features_valid)
        
        chunk_size = 50000
        n_valid = len(features_valid)
        preds_list = []
        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            chunk_pred = knn_model.predict(features_scaled[start:end])
            preds_list.append(chunk_pred)
            
        preds_all_valid = np.concatenate(preds_list)
        
        # 1. Segmentasi Morfologis Daratan Pulau
        daratan_inti = valid_mask & (ndvi > 0.15) & (mndwi < 0.10)
        daratan_morfologi = binary_closing(daratan_inti, structure=np.ones((7, 7)))
        daratan_morfologi = binary_fill_holes(daratan_morfologi)
        
        labeled_land, _ = label(daratan_morfologi)
        land_sizes = np.bincount(labeled_land.ravel())
        land_sizes[0] = 0
        
        mask_daratan_2d = np.isin(labeled_land, np.where(land_sizes > 100)[0])
        mask_daratan_2d = binary_dilation(mask_daratan_2d, structure=np.ones((5, 5))) & valid_mask
        
        mask_laut_murni_2d = valid_mask & ~mask_daratan_2d
        mask_laut_1d = mask_laut_murni_2d.flatten()[valid_mask_1d]
        mask_darat_1d = mask_daratan_2d.flatten()[valid_mask_1d]
        
        # Laut Lepas murni
        preds_all_valid[mask_laut_1d] = 'Laut'
        
        # 2. Danau di daratan
        mndwi_v = mndwi.flatten()[valid_mask_1d]
        b8_v = b8.flatten()[valid_mask_1d]
        ndvi_v = ndvi.flatten()[valid_mask_1d]
        
        is_air_daratan = mask_darat_1d & (mndwi_v > 0.12) & (b8_v < 1300)
        preds_all_valid[is_air_daratan] = 'Danau'
        
        # 3. Koreksi Biofisik Kawasan Timur (Jember, Banyuwangi, dsb.)
        false_bangunan = mask_darat_1d & (preds_all_valid == 'Bangunan') & (ndvi_v > 0.25)
        preds_all_valid[false_bangunan & (ndvi_v > 0.45)] = 'Hutan'
        preds_all_valid[false_bangunan & (ndvi_v <= 0.45)] = 'Sawah'
        
        salah_laut_di_darat = mask_darat_1d & (preds_all_valid == 'Laut')
        preds_all_valid[salah_laut_di_darat & (ndvi_v > 0.40)] = 'Hutan'
        preds_all_valid[salah_laut_di_darat & (ndvi_v <= 0.40) & (ndvi_v > 0.20)] = 'Sawah'
        preds_all_valid[salah_laut_di_darat & (ndvi_v <= 0.20)] = 'Bangunan'
        
        pred_1d[valid_mask_1d] = preds_all_valid

    pred_2d = pred_1d.reshape((height, width))
    valid_2d = valid_mask
    
    # Plotting ke Peta Folium
    m_full = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=8, 
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    for label_str, hex_color in colors.items():
        rgba_img = np.zeros((height, width, 4), dtype=np.uint8)
        mask = (pred_2d == label_str) & valid_2d
        
        if np.any(mask):
            r, g, b = hex_to_rgb(hex_color)
            rgba_img[mask, 0] = r
            rgba_img[mask, 1] = g
            rgba_img[mask, 2] = b
            rgba_img[mask, 3] = 180
            
            folium.raster_layers.ImageOverlay(
                image=rgba_img,
                bounds=img_bounds,
                opacity=0.7,
                name=f"Sebaran {label_str}",
                show=True
            ).add_to(m_full)
    
    folium.LayerControl().add_to(m_full)

m_full
```
