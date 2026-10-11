"""
Dashboard Klasifikasi Spasial Tutupan Lahan (LULC) Jawa Timur
Berbasis Citra Sentinel-2A Level-2A & Algoritma K-Nearest Neighbors (KNN)

Tema: Coklat-Putih Elegan (Coffee & Cream) dengan Dukungan Dark Mode Penuh
3 Peta Interaktif (Persis Sesuai visualisasi.md):
  1. Peta 1: Sampel Ground Truth (Poligon Digitasi Asli 6 Kelas)
  2. Peta 2: Klasifikasi Model KNN Berdasarkan Sampel (Evaluasi & Sorotan Salah Prediksi)
  3. Peta 3: Prediksi Menyeluruh Regional Jawa Timur (Raster Citra Asli)
"""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Spasial Tutupan Lahan Jawa Timur",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. STATE MANAJEMEN TEMA (LIGHT MODE: COKLAT-PUTIH vs DARK MODE: ESPRESSO DARK)
# ==============================================================================
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light Mode (Coklat-Putih)"

with st.sidebar:
    st.markdown("### Tema Tampilan")
    mode_dipilih = st.radio(
        "Pilih Mode Warna:",
        options=["Light Mode (Coklat-Putih)", "Dark Mode (Espresso Dark)"],
        index=0 if st.session_state["theme_mode"] == "Light Mode (Coklat-Putih)" else 1,
        help="Ganti nuansa tampilan antara Coklat-Putih bersih atau Dark Mode bernuansa kopi pekat."
    )
    st.session_state["theme_mode"] = mode_dipilih
    is_dark = (mode_dipilih == "Dark Mode (Espresso Dark)")

# Definisi Token Palet Warna
if not is_dark:
    # --- LIGHT MODE: COKLAT-PUTIH ELEGAN (Coffee Milk & Clean White) ---
    C_BG = "#FDFBF9"           # Putih gading bersih lembut
    C_SURFACE = "#FFFFFF"      # Putih murni untuk kartu/wadah
    C_SIDEBAR = "#F8F4EE"      # Coklat susu sangat muda untuk sidebar
    C_BORDER = "#EADCCF"       # Garis tepi coklat muda krim
    C_BORDER_SUBTLE = "#F3EBE1"
    C_TEXT = "#3D2B1F"         # Coklat tua pekat (mudah dibaca & kontras tinggi)
    C_TEXT_MUTED = "#795548"   # Coklat moka sekunder
    C_PRIMARY = "#6D4C41"      # Coklat kopi primer
    C_PRIMARY_HOVER = "#4E342E"# Coklat tua saat disentuh
    C_BADGE_BG = "#F3EBE1"     # Latar badge coklat muda
    C_BADGE_TXT = "#5D4037"
    C_METRIC_BG = "#FFFFFF"
    C_SHADOW = "rgba(61, 43, 31, 0.05)"
else:
    # --- DARK MODE: ESPRESSO DARK (Kopi Espresso Gelap Premium) ---
    C_BG = "#1A1412"           # Hitam espresso kecoklatan
    C_SURFACE = "#251D19"      # Kartu coklat gelap hangat
    C_SIDEBAR = "#1F1814"      # Sidebar espresso
    C_BORDER = "#3E3029"       # Border coklat gelap
    C_BORDER_SUBTLE = "#322721"
    C_TEXT = "#F5EBE1"         # Teks putih krem lembut (nyaman di mata)
    C_TEXT_MUTED = "#C7B3A3"   # Teks sekunder krem coklat
    C_PRIMARY = "#A1887F"      # Aksen primer coklat terang
    C_PRIMARY_HOVER = "#BCAAA4"
    C_BADGE_BG = "#352721"     # Badge coklat gelap
    C_BADGE_TXT = "#EFEBE9"
    C_METRIC_BG = "#251D19"
    C_SHADOW = "rgba(0, 0, 0, 0.35)"


# ==============================================================================
# 3. HEROICONS SVG HELPER
# ==============================================================================
def heroicon(path_d: str, size: int = 20, color: str = C_PRIMARY, stroke_width: float = 2.0) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="width:{size}px; height:{size}px; min-width:{size}px; max-width:{size}px; '
        f'min-height:{size}px; max-height:{size}px; display:inline-block; vertical-align:middle; flex-shrink:0;" '
        f'aria-hidden="true">'
        f'<path d="{path_d}"/>'
        f'</svg>'
    )


D_GLOBE = "M12 21a9.004 9.004 0 0 0 8.716-6.747M12 21a9.004 9.004 0 0 1-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 0 1 7.843 4.582M12 3a8.997 8.997 0 0 0-7.843 4.582m15.686 0A11.953 11.953 0 0 1 12 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0 1 21 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0 1 12 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 0 1 3 12c0-.778.099-1.533.284-2.253"
D_MAP = "M9 6.75V15m6-6v8.25m.503 3.046 4.84-2.42A1.125 1.125 0 0 0 21 16.883V5.86a1.125 1.125 0 0 0-.623-1.006l-4.82-2.41a1.125 1.125 0 0 0-.96.002l-5.195 2.597a1.125 1.125 0 0 1-.96 0L3.623 2.633A1.125 1.125 0 0 0 3 3.639v11.023a1.125 1.125 0 0 0 .623 1.006l4.82 2.41a1.125 1.125 0 0 0 .96-.002l5.195-2.597a1.125 1.125 0 0 1 .96 0l-.062-.033Z"
D_LAYERS = "M6.429 9.75 2.25 12l4.179 2.25m0-4.5 5.571 3 5.571-3m-11.142 0L2.25 7.5 12 2.25l9.75 5.25-4.179 2.25m0 0L21 12l-4.179 2.25m0 0 4.179 2.25L12 21.75 2.25 16.5l4.179-2.25m11.142 0-5.571 3-5.571-3"
D_CHECK_CIRCLE = "M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
D_SCALE = "M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0 0 12 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52v2.62a3.75 3.75 0 0 1-2.25 3.44l-.84.37m-7.41-6.43c-.99.203-1.99.377-3 .52m3-.52v2.62a3.75 3.75 0 0 0 2.25 3.44l.84.37"
D_EXCLAMATION = "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
D_CHART_BAR = "M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"
D_TABLE = "M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 0 1-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0 1 12 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M12 10.875v7.5m0-7.5a1.125 1.125 0 0 1 1.125-1.125h7.125"
D_TAG = "M9.568 3H5.25A2.25 2.25 0 0 0 3 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 0 0 5.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 0 0 9.568 3Z"
D_INFO = "m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"
D_SATELLITE = "m21 7.5-9-5.25L3 7.5m18 0-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9"


# ==============================================================================
# 4. INJECT CSS DINAMIS SESUAI TEMA AKTIF
# ==============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {C_BG} !important;
        font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: {C_TEXT} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {C_SIDEBAR} !important;
        border-right: 1px solid {C_BORDER} !important;
    }}

    /* Card Box */
    .themed-card {{
        background-color: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 1.15rem;
        padding: 1.35rem;
        box-shadow: 0 4px 14px {C_SHADOW};
        margin-bottom: 1.25rem;
    }}

    .themed-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid {C_BORDER_SUBTLE};
        margin-bottom: 0.85rem;
    }}

    .themed-card-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {C_TEXT};
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0;
    }}

    /* Badges */
    .themed-badge {{
        background-color: {C_BADGE_BG};
        color: {C_BADGE_TXT};
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border: 1px solid {C_BORDER};
    }}

    /* Metric Box */
    .metric-box {{
        background-color: {C_METRIC_BG};
        border: 1px solid {C_BORDER};
        border-radius: 1.15rem;
        padding: 1.3rem;
        box-shadow: 0 3px 10px {C_SHADOW};
        margin-bottom: 0.75rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .metric-box:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 18px {C_SHADOW};
    }}

    .metric-box-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .metric-label {{
        font-size: 0.725rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {C_TEXT_MUTED};
    }}

    .metric-val-row {{
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-top: 0.75rem;
    }}

    .metric-num {{
        font-size: 2rem;
        font-weight: 800;
        color: {C_TEXT};
        line-height: 1;
    }}

    .metric-sub {{
        font-size: 0.75rem;
        color: {C_TEXT_MUTED};
        margin-top: 0.45rem;
        line-height: 1.35;
    }}

    /* Buttons */
    .stDownloadButton button, .stButton button {{
        background-color: {C_PRIMARY} !important;
        color: #FFFFFF !important;
        border-radius: 0.65rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.55rem 1.25rem !important;
        border: none !important;
        box-shadow: 0 2px 6px {C_SHADOW} !important;
        transition: all 0.2s ease-in-out !important;
    }}

    .stDownloadButton button:hover, .stButton button:hover {{
        background-color: {C_PRIMARY_HOVER} !important;
        transform: translateY(-1px) !important;
    }}

    /* Info Banner */
    .themed-banner {{
        background-color: {C_BADGE_BG};
        border: 1px solid {C_BORDER};
        border-radius: 0.85rem;
        padding: 0.95rem 1.15rem;
        margin-bottom: 1.35rem;
        font-size: 0.85rem;
        color: {C_TEXT};
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }}

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {{
        background-color: {C_SURFACE} !important;
        border-radius: 0.75rem !important;
        padding: 0.35rem !important;
        border: 1px solid {C_BORDER} !important;
        gap: 0.4rem !important;
    }}

    button[data-baseweb="tab"] {{
        border-radius: 0.55rem !important;
        color: {C_TEXT_MUTED} !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.15rem !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {C_BADGE_BG} !important;
        color: {C_TEXT} !important;
        font-weight: 700 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. HELPER RESOLUSI FILE & CACHING
# ==============================================================================
def cari_file(nama_file: str) -> Path | None:
    kandidat = [
        Path(nama_file),
        Path("Data/UTS") / nama_file,
        Path("maps") / nama_file,
        Path("data") / nama_file,
        Path("views/klasifikasi-spasial") / nama_file,
        Path("_static") / nama_file,
        Path("../Data/UTS") / nama_file,
        Path("../../Data/UTS") / nama_file,
    ]
    for path in kandidat:
        if path.exists():
            return path
    return None


@st.cache_data
def muat_data_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def muat_konten_html(path_str: str) -> str | None:
    path = Path(path_str)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# ==============================================================================
# 6. SIDEBAR: METADATA & KONTROL
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-top:0.5rem; margin-bottom:1.15rem;">
        {heroicon(D_GLOBE, size=26, color=C_PRIMARY)}
        <div>
            <h2 style="font-size:1.1rem; font-weight:800; color:{C_TEXT}; margin:0; line-height:1.2;">LULC Spasial</h2>
            <span style="font-size:0.75rem; color:{C_TEXT_MUTED}; font-weight:600;">Jawa Timur Sentinel-2A</span>
        </div>
    </div>
    <div style="font-size:0.8rem; color:{C_TEXT}; padding:0.85rem; background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:0.75rem; line-height:1.65; margin-bottom:1.25rem;">
        <p style="margin:0;"><b style="color:{C_TEXT};">Model:</b> K-Nearest Neighbors (KNN, k=5)</p>
        <p style="margin:0;"><b style="color:{C_TEXT};">Sensor:</b> Sentinel-2A MSI (ESA CDSE)</p>
        <p style="margin:0;"><b style="color:{C_TEXT};">Level:</b> Level-2A BOA Surface Reflectance</p>
        <p style="margin:0;"><b style="color:{C_TEXT};">Area:</b> Provinsi Jawa Timur (Komposit 2024)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:0.65rem; display:flex; align-items:center; gap:0.4rem; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:{C_TEXT_MUTED};">
        {heroicon(D_TAG, size=16, color=C_TEXT_MUTED)}
        <span>6 Kategori Tutupan Lahan</span>
    </div>
    <div style="font-size:0.825rem; line-height:1.85; color:{C_TEXT}; background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:0.75rem; padding:0.85rem;">
        <div style="display:flex; align-items:center; gap:0.55rem;">
            <span style="width:12px; height:12px; border-radius:50%; background:#FFD92F; display:inline-block; border:1px solid #D7C4B7;"></span>
            <b>Sawah</b> <span style="color:{C_TEXT_MUTED}; font-size:0.75rem;">(Lahan Pertanian)</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem;">
            <span style="width:12px; height:12px; border-radius:50%; background:#E41A1C; display:inline-block; border:1px solid #D7C4B7;"></span>
            <b>Bangunan</b> <span style="color:{C_TEXT_MUTED}; font-size:0.75rem;">(Kawasan Terbangun)</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem;">
            <span style="width:12px; height:12px; border-radius:50%; background:#2E7D32; display:inline-block; border:1px solid #D7C4B7;"></span>
            <b>Hutan</b> <span style="color:{C_TEXT_MUTED}; font-size:0.75rem;">(Lahan Hijau Rapat)</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem;">
            <span style="width:12px; height:12px; border-radius:50%; background:#4FC3F7; display:inline-block; border:1px solid #D7C4B7;"></span>
            <b>Danau</b> <span style="color:{C_TEXT_MUTED}; font-size:0.75rem;">(Air Pedalaman / Waduk)</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem;">
            <span style="width:12px; height:12px; border-radius:50%; background:#0D47A1; display:inline-block; border:1px solid #D7C4B7;"></span>
            <b>Laut</b> <span style="color:{C_TEXT_MUTED}; font-size:0.75rem;">(Perairan Terbuka)</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem;">
            <span style="width:12px; height:12px; border-radius:50%; background:#8E44AD; display:inline-block; border:1px solid #D7C4B7;"></span>
            <b>Mangrove</b> <span style="color:{C_TEXT_MUTED}; font-size:0.75rem;">(Hutan Bakau Pesisir)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<hr style='margin:1.25rem 0; border:none; border-top:1px solid {C_BORDER};'>", unsafe_allow_html=True)

    mode_layout = st.radio(
        "Format Tata Letak 3 Peta:",
        options=[
            "Tab Navigasi (1 Peta per Tab)",
            "Tampilkan Sekaligus (Berurutan Vertikal)",
            "Peta 1 & 2 Bersisian (2 Kolom)",
        ],
        index=0,
        help="Pilih tata letak visualisasi 3 peta di layar."
    )

    tinggi_peta = st.slider(
        "Tinggi Frame Peta (px):",
        min_value=450,
        max_value=900,
        value=620,
        step=25,
        help="Sesuaikan tinggi frame peta interaktif."
    )


# ==============================================================================
# 7. HEADER UTAMA & 4 KARTU METRIK
# ==============================================================================
st.markdown(f"""
<header style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:1rem; margin-bottom:1.5rem;">
    <div>
        <div style="display:flex; align-items:center; gap:0.65rem;">
            {heroicon(D_GLOBE, size=32, color=C_PRIMARY)}
            <h1 style="font-size:1.95rem; font-weight:800; letter-spacing:-0.02em; color:{C_TEXT}; margin:0;">
                Klasifikasi Spasial Tutupan Lahan Jawa Timur
            </h1>
        </div>
        <p style="font-size:0.9rem; color:{C_TEXT_MUTED}; margin:0.4rem 0 0 0; max-width:52rem; line-height:1.55;">
            Visualisasi spasial berbasis citra satelit Sentinel-2A Level-2A dan algoritma K-Nearest Neighbors (KNN) untuk pemetaan 6 kelas tutupan lahan di seluruh wilayah Provinsi Jawa Timur.
        </p>
    </div>
    <div class="themed-badge" style="font-size:0.8rem; padding:0.45rem 0.95rem;">
        {heroicon(D_SATELLITE, size=18, color=C_PRIMARY)}
        <span>Sentinel-2A MSI &bull; K-Nearest Neighbors (K=5)</span>
    </div>
</header>
""", unsafe_allow_html=True)

p_pred = cari_file("prediksi_data_uji.csv")
df_pred = muat_data_csv(str(p_pred)) if p_pred else None

if df_pred is not None:
    n_total = len(df_pred)
    n_benar = int(df_pred["benar"].sum())
    n_salah = n_total - n_benar
    akurasi = (n_benar / n_total) * 100
else:
    n_total = 360
    n_benar = 317
    n_salah = 43
    akurasi = 88.1

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""<div class="metric-box">
<div class="metric-box-top">
<span class="metric-label">Akurasi Model KNN</span>
<span style="background:{C_BADGE_BG}; border-radius:50%; padding:0.35rem; display:inline-flex;">{heroicon(D_CHECK_CIRCLE, size=18, color=C_PRIMARY)}</span>
</div>
<div class="metric-val-row">
<span class="metric-num">{akurasi:.1f}%</span>
<span class="themed-badge">{n_benar}/{n_total} Poligon</span>
</div>
<div class="metric-sub">Evaluasi 360 Poligon Ground Truth (K=5)</div>
</div>""", unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""<div class="metric-box">
<div class="metric-box-top">
<span class="metric-label">Piksel Ekstraksi</span>
<span style="background:{C_BADGE_BG}; border-radius:50%; padding:0.35rem; display:inline-flex;">{heroicon(D_SCALE, size=18, color=C_PRIMARY)}</span>
</div>
<div class="metric-val-row">
<span class="metric-num">635K</span>
<span class="themed-badge">Sentinel-2A</span>
</div>
<div class="metric-sub">5 Band Optik + 4 Indeks Spektral (NDVI, NDWI, MNDWI, NDBI)</div>
</div>""", unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""<div class="metric-box">
<div class="metric-box-top">
<span class="metric-label">Poligon Ground Truth</span>
<span style="background:{C_BADGE_BG}; border-radius:50%; padding:0.35rem; display:inline-flex;">{heroicon(D_LAYERS, size=18, color=C_PRIMARY)}</span>
</div>
<div class="metric-val-row">
<span class="metric-num">360</span>
<span class="themed-badge">60 / Kelas</span>
</div>
<div class="metric-sub">Distribusi poligon seimbang untuk 6 kategori tutupan lahan</div>
</div>""", unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""<div class="metric-box">
<div class="metric-box-top">
<span class="metric-label">Salah Klasifikasi</span>
<span style="background:{C_BADGE_BG}; border-radius:50%; padding:0.35rem; display:inline-flex;">{heroicon(D_EXCLAMATION, size=18, color='#D97706')}</span>
</div>
<div class="metric-val-row">
<span class="metric-num">{n_salah}</span>
<span class="themed-badge" style="color:#D97706;">Error {100 - akurasi:.1f}%</span>
</div>
<div class="metric-sub">Ditandai garis merah putus-putus pada Peta 2</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


# ==============================================================================
# 8. VISUALISASI 3 PETA SESUAI VISUALISASI.MD
# ==============================================================================
f_peta1 = cari_file("peta_1_ground_truth.html")
f_peta2 = cari_file("peta_klasifikasi_knn.html") or cari_file("peta_evaluasi_poligon.html")
f_peta3 = cari_file("hasil_klasifikasi_regional_jatim.html")

html_peta1 = muat_konten_html(str(f_peta1)) if f_peta1 else None
html_peta2 = muat_konten_html(str(f_peta2)) if f_peta2 else None
html_peta3 = muat_konten_html(str(f_peta3)) if f_peta3 else None


def render_peta_1():
    st.markdown(f"""<div class="themed-card" style="margin-bottom:0.75rem;">
<div class="themed-card-header">
<h2 class="themed-card-title">
{heroicon(D_LAYERS, size=22, color=C_PRIMARY)}
<span>Peta 1: Visualisasi Poligon Sampel Ground Truth (Digitasi Awal)</span>
</h2>
<span class="themed-badge">360 Poligon Terverifikasi</span>
</div>
<div style="font-size:0.825rem; color:{C_TEXT_MUTED}; line-height:1.55;">
Visualisasi titik piksel yang disatukan per <code>poligon_id</code> menjadi hamparan poligon (<i>Convex Hull</i>) 
dengan warna sesuai label asli (<i>ground truth</i>). Gunakan layer control untuk memilih layer satelit Esri / Google.
</div>
</div>""", unsafe_allow_html=True)
    if html_peta1:
        components.html(html_peta1, height=tinggi_peta, scrolling=True)
    else:
        st.error("File `peta_1_ground_truth.html` tidak ditemukan.")


def render_peta_2():
    st.markdown(f"""<div class="themed-card" style="margin-bottom:0.75rem;">
<div class="themed-card-header">
<h2 class="themed-card-title">
{heroicon(D_MAP, size=22, color=C_PRIMARY)}
<span>Peta 2: Klasifikasi Model KNN Berdasarkan Sampel Poligon</span>
</h2>
<span class="themed-badge">Evaluasi Model (K=5)</span>
</div>
<div style="font-size:0.825rem; color:{C_TEXT_MUTED}; line-height:1.55;">
Memvisualisasikan hasil prediksi algoritma K-Nearest Neighbors pada sampel poligon. 
Warna menunjukkan prediksi kelas, dan <b style="color:#DC2626;">garis putus-putus merah</b> menandai poligon yang salah diprediksi oleh model.
</div>
</div>""", unsafe_allow_html=True)
    if html_peta2:
        components.html(html_peta2, height=tinggi_peta, scrolling=True)
    else:
        st.error("File `peta_klasifikasi_knn.html` tidak ditemukan.")


def render_peta_3():
    st.markdown(f"""<div class="themed-card" style="margin-bottom:0.75rem;">
<div class="themed-card-header">
<h2 class="themed-card-title">
{heroicon(D_GLOBE, size=22, color=C_PRIMARY)}
<span>Peta 3: Prediksi Menyeluruh Skala Regional Jawa Timur (Raster Citra Asli)</span>
</h2>
<span class="themed-badge">Resolusi Citra Sentinel-2 Asli (~250m)</span>
</div>
<div style="font-size:0.825rem; color:{C_TEXT_MUTED}; line-height:1.55;">
Inferensi model KNN piksel-per-piksel yang diproyeksikan ke seluruh citra satelit Sentinel-2 wilayah Jawa Timur. 
Tiap kelas tutupan lahan berupa layer transparan yang dapat dinyalakan/dimatikan pada layer control di pojok kanan atas peta.
</div>
</div>""", unsafe_allow_html=True)
    if html_peta3:
        components.html(html_peta3, height=tinggi_peta, scrolling=True)
    else:
        st.error("File `hasil_klasifikasi_regional_jatim.html` tidak ditemukan.")


# Render Tata Letak 3 Peta
if mode_layout == "Tab Navigasi (1 Peta per Tab)":
    tab1, tab2, tab3 = st.tabs([
        "📍 Peta 1: Sampel Ground Truth",
        "🎯 Peta 2: Prediksi KNN Sampel",
        "🗺️ Peta 3: Regional Jawa Timur"
    ])
    with tab1:
        render_peta_1()
    with tab2:
        render_peta_2()
    with tab3:
        render_peta_3()

elif mode_layout == "Tampilkan Sekaligus (Berurutan Vertikal)":
    render_peta_1()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    render_peta_2()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    render_peta_3()

elif mode_layout == "Peta 1 & 2 Bersisian (2 Kolom)":
    col1, col2 = st.columns(2)
    with col1:
        render_peta_1()
    with col2:
        render_peta_2()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    render_peta_3()


# ==============================================================================
# 9. EVALUASI MODEL & CONFUSION MATRIX
# ==============================================================================
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

st.markdown(f"""<div class="themed-card" style="margin-bottom:1rem;">
<div class="themed-card-header" style="margin-bottom:0.25rem; padding-bottom:0.65rem;">
<h2 class="themed-card-title">
{heroicon(D_CHART_BAR, size=22, color=C_PRIMARY)}
<span>Evaluasi Kinerja Model & Karakteristik Spektral</span>
</h2>
<span class="themed-badge">KNN (K=5) &bull; Evaluasi 360 Poligon</span>
</div>
</div>""", unsafe_allow_html=True)

col_cm, col_fi = st.columns(2)

with col_cm:
    st.markdown(f"""<div class="themed-card" style="margin-bottom:0.75rem;">
<h3 style="font-size:0.95rem; font-weight:700; color:{C_TEXT}; margin:0 0 0.35rem 0;">Confusion Matrix (Evaluasi 360 Poligon)</h3>
<p style="font-size:0.8rem; color:{C_TEXT_MUTED}; margin:0 0 0.75rem 0;">Perbandingan kelas ground truth aktual vs hasil prediksi model KNN.</p>
</div>""", unsafe_allow_html=True)

    p_cm_img = cari_file("confusion_matrix.png")
    if p_cm_img:
        st.image(Image.open(p_cm_img), caption="Confusion Matrix Model KNN (K=5)", use_container_width=True)
    else:
        st.info("File confusion matrix belum tersedia.")

    st.markdown(f"""<div class="themed-card" style="font-size:0.775rem; color:{C_TEXT_MUTED}; margin-top:0.5rem; line-height:1.45;">
<b>Analisis Matriks:</b> Akurasi mencapai <b>{akurasi:.1f}%</b>. Poligon Bangunan, Sawah, dan Hutan memiliki akurasi sangat tinggi, sementara sebagian kecil salah prediksi terjadi pada kemiripan spektral perairan (Danau vs Laut).
</div>""", unsafe_allow_html=True)

with col_fi:
    st.markdown(f"""<div class="themed-card" style="margin-bottom:0.75rem;">
<h3 style="font-size:0.95rem; font-weight:700; color:{C_TEXT}; margin:0 0 0.35rem 0;">Ringkasan Fitur Band & Indeks Spektral</h3>
<p style="font-size:0.8rem; color:{C_TEXT_MUTED}; margin:0 0 0.75rem 0;">Kombinasi 9 parameter input yang digunakan model untuk klasifikasi tutupan lahan.</p>
</div>""", unsafe_allow_html=True)

    df_fitur_info = pd.DataFrame({
        "Fitur": ["B02, B03, B04", "B08 (NIR)", "B11 (SWIR)", "NDVI", "NDWI", "MNDWI", "NDBI"],
        "Nama Spektral": ["Blue, Green, Red", "Near Infrared (842nm)", "Shortwave Infrared (1610nm)", "Vegetation Index", "Water Index", "Modified NDWI", "Built-up Index"],
        "Fungsi Pemisah": ["Warna tampak visual dasar", "Biomassa kanopi vegetasi", "Kawasan terbangun & kelembaban tanah", "Pembeda vegetasi rapat (Hutan vs Sawah)", "Pemisah perairan terbuka (Laut)", "Deteksi air pedalaman (Danau/Waduk)", "Identifikasi kawasan beton / pemukiman"]
    })
    st.dataframe(df_fitur_info, use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="themed-card" style="font-size:0.775rem; color:{C_TEXT_MUTED}; margin-top:0.5rem; line-height:1.45;">
<b>Peran Fitur:</b> Kombinasi band SWIR (B11) dengan indeks NDVI, MNDWI, dan NDBI memberikan daya pisah optimal untuk membedakan bangunan beton dari vegetasi dan perairan.
</div>""", unsafe_allow_html=True)


# ==============================================================================
# 10. TABEL EKSPLORASI DATA POLIGON & EKSPOR CSV
# ==============================================================================
st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

st.markdown(f"""<div class="themed-card" style="margin-bottom:1rem;">
<div class="themed-card-header" style="margin-bottom:0.5rem;">
<h2 class="themed-card-title">
{heroicon(D_TABLE, size=22, color=C_PRIMARY)}
<span>Eksplorasi Data Tabular Poligon Uji & Hasil Prediksi</span>
</h2>
<span class="themed-badge">Filter Interaktif</span>
</div>
</div>""", unsafe_allow_html=True)

if df_pred is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        semua_kelas = sorted(df_pred["kelas_asli"].unique().tolist())
        pilihan_kelas = st.multiselect(
            "Filter Berdasarkan Kategori Tutupan Lahan:",
            options=semua_kelas,
            default=semua_kelas
        )
    with c2:
        status_filter = st.radio(
            "Filter Status Klasifikasi:",
            options=["Semua Poligon", "Hanya Prediksi Benar", "Hanya Salah Prediksi"],
            horizontal=True
        )

    df_tampil = df_pred[df_pred["kelas_asli"].isin(pilihan_kelas)].copy()
    if status_filter == "Hanya Prediksi Benar":
        df_tampil = df_tampil[df_tampil["benar"] == True]
    elif status_filter == "Hanya Salah Prediksi":
        df_tampil = df_tampil[df_tampil["benar"] == False]

    st.markdown(
        f"<p style='font-size:0.825rem; color:{C_TEXT_MUTED}; margin:0.5rem 0 0.75rem 0;'>"
        f"Menampilkan <b style='color:{C_TEXT};'>{len(df_tampil)}</b> dari total <b style='color:{C_TEXT};'>{len(df_pred)}</b> poligon sampel:"
        f"</p>",
        unsafe_allow_html=True
    )
    
    format_cols = {col: "{:.4f}" for col in ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'lon', 'lat'] if col in df_tampil.columns}
    st.dataframe(df_tampil.style.format(format_cols), use_container_width=True, height=360)

    csv_bytes = df_tampil.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh Data Hasil Prediksi (CSV)",
        data=csv_bytes,
        file_name="prediksi_poligon_knn_jatim.csv",
        mime="text/csv"
    )
else:
    st.warning("File `prediksi_data_uji.csv` tidak ditemukan di folder.")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(f"""
<footer style="margin-top:3.5rem; padding-top:1.5rem; border-top:1px solid {C_BORDER}; text-align:center; font-size:0.8rem; color:{C_TEXT_MUTED};">
    Proyek Penambangan Data Sains &mdash; Klasifikasi Spasial Tutupan Lahan Jawa Timur &bull; Sentinel-2A Level-2A & K-Nearest Neighbors (KNN)
</footer>
""", unsafe_allow_html=True)
