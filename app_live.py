"""SAM3 - Etkileşimli Bina Çatısı Tespit Paneli - ana giriş noktası.

    pipeline/config.py            - sabitler + DetectionSettings
    pipeline/geo_utils.py          - tile matematiği, canlı indirme
    pipeline/tif_upload.py          - TIF önizleme/kırpma/referans onarımı
    pipeline/mask_utils.py           - maske/poligon yardımcıları
    pipeline/models.py                - model yükleme
    pipeline/detection.py              - run_detection_pipeline() (asıl boru hattı)
    pipeline/ui_sidebar.py              - sol kenar çubuğu
    pipeline/ui_map_section.py           - harita ile tespit bölümü
    pipeline/ui_tif_section.py            - TIF ile tespit bölümü
    pipeline/ui_results_section.py         - sonuç/metrik gösterimi

"""
import os

import streamlit as st
from dotenv import load_dotenv

from pipeline.models import find_latest_yolo_path, load_models
from pipeline.ui_map_section import render_map_section
from pipeline.ui_results_section import render_results_section
from pipeline.ui_sidebar import render_sidebar
from pipeline.ui_tif_section import render_tif_section

# .env dosyasından GOOGLE_MAPS_API_KEY oku
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

st.set_page_config(layout="wide", page_title="SAM3 - Canlı Bina Çatısı Tespiti")

st.title("🛰️ SAM3 - Etkileşimli Bina Çatısı Tespit Paneli")
st.caption("Haritada istediğiniz bölgeye yakınlaşın, dikdörtgen aracıyla alan seçin ve tespiti çalıştırın.")

# --- SIDEBAR ---
settings, map_source, mask_color = render_sidebar(GOOGLE_MAPS_API_KEY)

# --- MODELLER ---
YOLO_MODEL_PATH = find_latest_yolo_path()

if YOLO_MODEL_PATH:
    st.sidebar.success(f"✅ YOLO Modeli Bulundu:\n`{os.path.basename(os.path.dirname(os.path.dirname(YOLO_MODEL_PATH)))}`")
else:
    st.sidebar.error("⏳ Eğitilmiş YOLO ağırlığı bulunamadı!")

yolo_model, sam = (None, None)
if YOLO_MODEL_PATH:
    with st.spinner("Modeller yükleniyor..."):
        yolo_model, sam = load_models(YOLO_MODEL_PATH)

# --- 1. HARİTA İLE TESPİT ---
active_source = render_map_section(settings, map_source, GOOGLE_MAPS_API_KEY, yolo_model, sam)

# --- 2. TIF DOSYASINDAN TESPİT (HARİTADAN AYRI) ---
render_tif_section(settings, yolo_model, sam)

# --- 3. SONUÇLAR (ARA AŞAMA GÖRSELLERİ + TESPİT SONUCU + METRİKLER) ---
render_results_section(settings, active_source, GOOGLE_MAPS_API_KEY, mask_color)
