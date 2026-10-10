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

# 2. Data Understanding & Karakteristik Data Spasial

## 2.1. Definisi Domain dan 6 Kelas Tutupan Lahan
Area of Interest (AOI) mencakup seluruh wilayah daratan dan perairan pesisir Provinsi Jawa Timur. Objek permukaan bumi dikelompokkan ke dalam 6 kelas tutupan lahan utama:

| No | Nama Kelas | Kode Label | Definisi & Karakteristik Objek | Respon Spektral Utama |
|---|---|---|---|---|
| 1 | Sawah | 1 | Lahan pertanian padi basah berpetak. Kondisi bervariasi dari genangan air awal tanam, fase vegetatif hijau pekat, hingga fase pematangan/panen. | Pantulan NIR sedang-tinggi pada fase vegetatif; nilai SWIR meningkat pada lahan bera/kering. |
| 2 | Bangunan | 2 | Kawasan terbangun perkotaan, permukiman, jalan aspal/beton, dan atap bangunan (genteng/seng). | Pantulan tinggi pada band SWIR (B11) dan Red (B04); nilai NDBI positif tinggi; NDVI rendah. |
| 3 | Hutan (Lahan Hijau)| 3 | Vegetasi alami kanopi pohon lebat, hutan pegunungan, perbukitan hijau, dan perkebunan tahunan rapat. | Pantulan sangat tinggi pada NIR (B08) akibat struktur sel daun; serapan kuat pada Red (B04); NDVI sangat tinggi. |
| 4 | Danau | 4 | Badan air tawar pedalaman (danau alami, waduk, telaga, bendungan) berarus tenang. | Penyerapan kuat pada inframerah (NIR dan SWIR); nilai NDWI dan MNDWI sangat positif; NDVI negatif. |
| 5 | Laut (Perairan Terbuka) | 5 | Badan air asin laut terbuka di pesisir utara dan selatan Jawa Timur serta Selat Madura. | Pantulan dominan pada band Blue (B02) dan Green (B03); penyerapan hampir total pada NIR/SWIR; NDWI tinggi. |
| 6 | Mangrove | 6 | Komunitas hutan bakau pesisir yang toleran terhadap air asin/payau, tumbuh di muara sungai dan pantai berlumpur. | Kombinasi sinyal vegetasi lebat (NIR tinggi) dan pengaruh substrat air/lumpur basah (SWIR rendah, MNDWI relatif tinggi dibanding hutan daratan). |

## 2.2. Karakteristik Citra Satelit Sentinel-2A Level-2A
Dataset citra satelit diperoleh dari satelit optik Sentinel-2A Level-2A (Bottom-of-Atmosphere / BOA Surface Reflectance) yang disediakan oleh European Space Agency (ESA) melalui Copernicus Data Space Ecosystem (CDSE). Akuisisi data dilakukan untuk rentang waktu **1 September 2026 – 30 September 2026** dengan ambang batas tutupan awan (cloud cover) maksimal 30%.

Band spektral yang digunakan meliputi:
*   **B02 (Blue - 490 nm):** Resolusi spasial 10 meter. Sensitif terhadap kedalaman air dan hamburan atmosfer.
*   **B03 (Green - 560 nm):** Resolusi spasial 10 meter. Puncak pantulan klorofil daun pada spektrum tampak dan deteksi air bersih.
*   **B04 (Red - 665 nm):** Resolusi spasial 10 meter. Pita serapan utama klorofil; sangat berguna membedakan vegetasi hidup dari tanah terbuka atau bangunan.
*   **B08 (Near Infrared / NIR - 842 nm):** Resolusi spasial 10 meter. Pantulan maksimal struktur kanopi sel mesofil daun vegetasi; diserap hampir sempurna oleh badan air jernih.
*   **B11 (Short-Wave Infrared / SWIR - 1610 nm):** Resolusi spasial 20 meter (di-resample ke 10 m). Sensitif terhadap kadar kelembapan tanah, kebasahan vegetasi, dan material semen/bebatuan bangunan.
*   **SCL (Scene Classification Layer):** Lapisan klasifikasi kualitas piksel otomatis untuk memfilter piksel yang mengandung awan (tebal, sedang, cirrus), bayangan awan (*cloud shadow*), serta piksel jenuh (defective).

## 2.3. Rekayasa Fitur Indeks Spektral (Spectral Indices)
Untuk mempertegas batas pemisah antar-kelas (*feature separability*), dihitung empat indeks spektral turunan yang dinormalisasi:

1.  **Normalized Difference Vegetation Index (NDVI)**
    Membedakan kerapatan dan kehijauan vegetasi dari non-vegetasi:
    $$NDVI = \frac{B08 - B04}{B08 + B04 + \epsilon}$$
2.  **Normalized Difference Water Index (NDWI - McFeeters)**
    Mengisolasi badan air permukaan terbuka dari tutupan lahan daratan:
    $$NDWI = \frac{B03 - B08}{B03 + B08 + \epsilon}$$
3.  **Modified Normalized Difference Water Index (MNDWI - Xu)**
    Meningkatkan akurasi deteksi air dengan menekan sinyal pantulan area terbangun/bangunan:
    $$MNDWI = \frac{B03 - B11}{B03 + B11 + \epsilon}$$
4.  **Normalized Difference Built-up Index (NDBI)**
    Mendeteksi konsentrasi kawasan terbangun, perkerasan beton, dan lahan terbuka:
    $$NDBI = \frac{B11 - B08}{B11 + B08 + \epsilon}$$

*(Nilai $\epsilon = 1 \times 10^{-9}$ ditambahkan pada penyebut untuk mencegah kesalahan pembagian dengan nol / division by zero).*

## 2.4. Sebaran Data Ground Truth dan Hasil Ekstraksi
Data *ground truth* dikumpulkan dalam bentuk poligon Shapefile yang didigitasi secara akurat dari citra resolusi tinggi. Dari keseluruhan proses validasi dan penyaringan perpotongan spasial (*intersection*), terdapat **274 poligon sampel valid**. Mengingat luasnya beberapa poligon (terutama laut dan hutan), dilakukan pembatasan maksimal **2.000 piksel per poligon** menggunakan subsampling acak agar distribusi antar-kelas tetap seimbang (*seed* pengacakan = 42).

Dari poligon tersebut, berhasil diekstraksi sebanyak **322.146 piksel berlabel** dengan distribusi per kelas sebagai berikut:
*   **Laut:** 92.105 piksel
*   **Sawah:** 67.940 piksel
*   **Danau:** 49.850 piksel
*   **Mangrove:** 45.120 piksel
*   **Hutan:** 38.719 piksel
*   **Bangunan:** 28.412 piksel

---

# Metodologi Pengambilan Data (Data Acquisition via openEO)
Wilayah Provinsi Jawa Timur membentang sangat luas. Mengunduh citra satelit sekaligus dalam satu *scene* besar akan menyebabkan kendala *out-of-memory*, batas kuota pemrosesan (*timeout*), dan ukuran file yang terlampau besar. 

Oleh karena itu, diterapkan strategi komputasi berikut dalam akuisisinya:

1.  **Spatial Grid Tiling:** Wilayah studi dibagi ke dalam blok-blok grid berukuran 0,20° (~22 km × 22 km) dengan tambahan margin *buffer* sebesar 0,005° (~500 m) agar poligon yang berada di perbatasan sel tetap terakuisisi secara utuh.
2.  **Parallel Batch Job Processing:** Pengiriman *batch job* asinkron ke server openEO CDSE dikelola menggunakan `MultiBackendJobManager` secara paralel (2 *worker* konkuren) dilengkapi mekanisme retri otomatis apabila terjadi kegagalan.
3.  **Pembersihan Awan Tingkat Piksel:** Memanfaatkan band `SCL` (Scene Classification Layer) untuk melakukan *masking* atau membuang piksel awan dan bayangan awan sebelum dilakukan agregasi temporal.
4.  **Agregasi Temporal:** Dilakukan komposit temporal menggunakan operasi `median_time()` atas seluruh *scene* yang tersedia sepanjang bulan September 2026. Hal ini menghasilkan satu lapisan citra bebas awan dengan nilai piksel representatif.
5.  **Deduplikasi dan Penugasan Sel Grid:** Hanya piksel yang pusat koordinatnya benar-benar berada di dalam sel grid (tanpa *buffer*) yang disimpan. Hal ini mencegah terjadinya duplikasi ekstraksi piksel pada area yang saling tumpang tindih (*overlap*) antar-tile.
6.  **Kode Program Ekstraksi:** Berikut merupakan skrip Python untuk melakukan ekstraksi dan penarikan data tersebut. Konfigurasi direktori *output* telah disesuaikan agar tersimpan di direktori `../../Data/UTS/` (atau sebagai file `piksel_s2_zaenal.csv`).

```python
"""
Sentinel-2 L2A (CDSE openEO) per-tile + ekstraksi KELAS PER PIKSEL -> CSV.
Setiap baris CSV = satu piksel dengan atribut band spektral dan indeks turunan.
"""

from datetime import datetime
from pathlib import Path
import threading
import time

import geopandas as gpd
import numpy as np
import openeo
from openeo.extra.job_management import CsvJobDatabase, MultiBackendJobManager
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from shapely.geometry import box
from shapely.validation import make_valid

# ==============================================================================
# KONFIGURASI
# ==============================================================================
SUMBER = {  # nama kelas: (path file, kode label)
    "Sawah": ("sawah.zip", 1),
    "Bangunan": ("bangunan.zip", 2),
    "Hutan": ("Hutan.zip", 3),
    "Danau": ("Danau.zip", 4),
    "Laut": ("laut2.zip", 5),
    "Mangrove": ("Mangrove.zip", 6),
}

TILE_DEG = 0.20  # Ukuran tile ~22 km (~2200 x 2200 piksel @10m)
BUFFER_DEG = 0.005  # Margin ~500 m di sekeliling sampel

TANGGAL = ["2026-09-01", "2026-09-30"]
BANDS = ["B02", "B03", "B04", "B08", "B11"]
PAKAI_MASK_SCL = True  # Masking awan & bayangan berbasis SCL
MAX_CLOUD = 30  # Ambang batas tutupan awan scene

PARALEL = 2  # Jumlah worker paralel openEO CDSE
MAKS_PIKSEL_PER_POLIGON = 2000  # Sub-sampling acak poligon besar
SEED = 42
BUAT_RASTER_LABEL = True

OUT_DIR = Path("../../Data/UTS")
JOB_DB_PATH = OUT_DIR / "daftar_job.csv"
PETA_CSV = OUT_DIR / "peta_sampel_tile.csv"
PIKSEL_OUT = OUT_DIR / "piksel_s2_zaenal.csv"


# 1. BACA & GABUNGKAN POLIGON GROUND TRUTH
def baca_semua() -> gpd.GeoDataFrame:
  bagian = []
  for nama, (file, label) in SUMBER.items():
    g = gpd.read_file(file).to_crs("EPSG:4326")
    g["poligon_no"] = np.arange(len(g))
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.apply(
        lambda x: x if x.is_valid else make_valid(x)
    )
    g["label_teks"] = nama
    g["label"] = label
    g["poligon_id"] = [f"{nama}_{n:04d}" for n in g["poligon_no"]]
    g = g[["poligon_id", "poligon_no", "label", "label_teks", "geometry"]]
    print(
        f"Jumlah sampel {nama:<9}: {len(g)}  bounds={[round(v, 3) for v in g.total_bounds]}"
    )
    bagian.append(g)

  gdf = gpd.GeoDataFrame(pd.concat(bagian, ignore_index=True), crs="EPSG:4326")
  print(f"Total sampel gabungan   : {len(gdf)}")
  return gdf


# 2. PEMBAGIAN GRID TILING
def buat_tile(gdf: gpd.GeoDataFrame):
  T, B = TILE_DEG, BUFFER_DEG
  peta, batas, sel_xy = [], {}, {}

  for idx, geom in gdf.geometry.items():
    minx, miny, maxx, maxy = geom.bounds
    for gx in range(int(np.floor(minx / T)), int(np.floor(maxx / T)) + 1):
      for gy in range(int(np.floor(miny / T)), int(np.floor(maxy / T)) + 1):
        sel = box(gx * T, gy * T, (gx + 1) * T, (gy + 1) * T)
        if not geom.intersects(sel):
          continue
        bagian = geom.intersection(sel)
        if bagian.is_empty:
          continue
        b = bagian.bounds
        tid = f"x{gx}_y{gy}"
        peta.append({
            "tile_id": tid,
            "sampel_id": idx,
            "poligon_id": gdf.at[idx, "poligon_id"],
        })
        sel_xy[tid] = (gx, gy)
        if tid in batas:
          o = batas[tid]
          batas[tid] = (
              min(o[0], b[0]),
              min(o[1], b[1]),
              max(o[2], b[2]),
              max(o[3], b[3]),
          )
        else:
          batas[tid] = b

  tiles = pd.DataFrame([{
      "tile_id": tid,
      "gx": sel_xy[tid][0],
      "gy": sel_xy[tid][1],
      "west": b[0] - B,
      "south": b[1] - B,
      "east": b[2] + B,
      "north": b[3] + B,
  } for tid, b in batas.items()])
  peta = pd.DataFrame(peta)
  tiles["n_sampel"] = tiles["tile_id"].map(
      peta.groupby("tile_id")["sampel_id"].nunique()
  )
  return tiles, peta


# 3. KIRIM JOB KE OPENEO
def start_job(row, connection, **kwargs):
  bbox = {
      "west": float(row["west"]),
      "south": float(row["south"]),
      "east": float(row["east"]),
      "north": float(row["north"]),
  }
  bands = BANDS + ["SCL"] if PAKAI_MASK_SCL else BANDS
  cube = connection.load_collection(
      "SENTINEL2_L2A",
      spatial_extent=bbox,
      temporal_extent=TANGGAL,
      bands=bands,
      max_cloud_cover=MAX_CLOUD,
  )

  if PAKAI_MASK_SCL:
    scl = cube.band("SCL")
    # Mask: 1=jenuh, 3=bayangan awan, 8/9=awan sedang/tinggi, 10=cirrus
    mask_awan = (
        (scl == 1) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    )
    cube = cube.filter_bands(BANDS).mask(mask_awan)

  komposit = cube.median_time()
  return komposit.create_job(title=f"s2_{row['tile_id']}", out_format="GTiff")


# 4. TAMBAH INDEKS SPEKTRAL
def tambah_indeks(df: pd.DataFrame) -> pd.DataFrame:
  e = 1e-9
  df["NDVI"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"] + e)
  df["NDWI"] = (df["B03"] - df["B08"]) / (df["B03"] + df["B08"] + e)
  df["MNDWI"] = (df["B03"] - df["B11"]) / (df["B03"] + df["B11"] + e)
  df["NDBI"] = (df["B11"] - df["B08"]) / (df["B11"] + df["B08"] + e)
  return df
```

---

# 3. Exploratory Data Analysis (EDA)

Mari kita pahami karakteristik dari data yang telah dikumpulkan. Kita akan memuat dataset dan mengeksplorasi strukturnya.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Memuat data
df = pd.read_csv('../../Data/UTS/piksel_s2_zaenal.csv')

colors = {
    'Sawah': '#FFD92F',
    'Bangunan': '#E41A1C',
    'Mangrove': '#8E44AD',
    'Hutan': '#2E7D32',
    'Laut': '#0D47A1',
    'Danau': '#4FC3F7'
}

print(f"Total piksel yang diolah: {df.shape[0]}")
print(f"Banyaknya wilayah poligon unik: {df['poligon_id'].nunique()}")
df.head()
```

### Distribusi Kelas Lahan
Penting untuk mengetahui apakah proporsi data seimbang antarkelas.

```{code-cell} ipython3
plt.figure(figsize=(10, 5))
sns.countplot(y='label_teks', data=df, palette=colors, order=df['label_teks'].value_counts().index)
plt.title('Distribusi Jumlah Piksel per Kelas Tutupan Lahan')
plt.xlabel('Jumlah Sampel Piksel')
plt.ylabel('Kelas')
plt.show()
```

### Karakteristik Kerapatan Vegetasi (NDVI)
NDVI adalah indeks yang sangat baik untuk memisahkan vegetasi dengan daratan terbuka atau air.

```{code-cell} ipython3
plt.figure(figsize=(10, 6))
sns.boxplot(x='label_teks', y='NDVI', data=df, palette=colors)
plt.title('Distribusi Nilai NDVI Berdasarkan Kelas')
plt.xticks(rotation=45)
plt.show()
```
*Interpretasi:* Dapat dilihat bahwa kelas `Hutan` dan `Sawah` umumnya memiliki nilai NDVI yang jauh lebih tinggi (mendekati 1.0) dibandingkan kelas perairan seperti `Laut` dan `Danau` (bernilai negatif). Ini akan menjadi prediktor yang kuat untuk model kita.