"""Sabitler ve tespit ayarlarını tasiyan DetectionSettings veri sinifi.

run_detection_pipeline() ve tum modeller/hesaplamalar bu dosyadaki sabitlere
gore calisir. Sidebar widget'lari (ui_sidebar.py) bir DetectionSettings
nesnesi doldurup doner; run_detection_pipeline() artik sidebar'in kendi
degiskenlerine (global/closure) degil, bu nesneye bakar - boylece farkli
dosyalara bolunebiliyor.
"""
import math
import os
from dataclasses import dataclass


# --- Model / dosya yollari ---
RUNS_DIR = r"C:\1.Projeler\yolo-sam\runs\detect"

# shadow_unet_v3 - egitim ciktisi
SHADOW_UNET_DIR = r"D:\shadow-eo\runs\shadow_unet"
SHADOW_CHECKPOINT = os.path.join(SHADOW_UNET_DIR, "best.pt")

# --- Sabit (widget olmayan) parametreler ---
CONFIDENCE_THRESHOLD = 0.2  # YOLO confidence esigi 
ZOOM_LEVEL = 19             # Indirme zoom seviyesi 
REMOVE_SHADOWS = True       # Basit parlaklik esigi her tespitte otomatik uygulanir
BRIGHTNESS_THRESHOLD = 90
OPACITY = 0.25              # Sonuc haritasindaki poligon dolgu opakligi

# --- Tile / coğrafya sabitleri ---
TILE_SIZE = 256
EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = 2 * math.pi * EARTH_RADIUS / 2.0
SAM3_MIN_MASK_PX = 40

# --- Maske / poligon esikleri ---
MIN_POLYGON_AREA_M2 = 45.0
MAX_POLYGON_AREA_M2 = 1200.0  # Devasa zeminleri (park, ray, acik otopark) elemek icin ust sinir
MASK_MERGE_OVERLAP_THRESHOLD = 0.35

# --- Hazir konumlar (harita "Git" kisayollari) ---
PRESET_LOCATIONS = {
    "Muratpaşa(Antalya)": [36.8850, 30.7056],
    "Çankaya (Ankara)": [39.9179, 32.8627],
    "Torbalı (İzmir)": [38.1600, 27.3600],
    "Seferihisar (İzmir)": [38.1967, 26.8339],
    "Esenyurt (İstanbul)": [41.0300, 28.6800],
    "Çayyolu (Ankara)": [39.8745, 32.6889],
    "Beypazarı (Ankara)": [40.1667, 31.9167],
    "Fethiye (Muğla)": [36.6217, 29.1164],
    "Osmangazi (Bursa)": [40.1826, 29.0665],
    "Misi Köyü (Bursa)": [40.2100, 29.0980],
}


@dataclass
class DetectionSettings:
    """Sidebar'da toplanan ve run_detection_pipeline()'a geçirilen tüm ayarlar.

    ui_sidebar.render_sidebar() bunu doldurup döner; detection.run_detection_pipeline()
    sidebar'daki widget değişkenlerine değil, doğrudan bu nesneye bakar.
    """
    # Model & Parametreler
    nms_threshold: float
    edge_buffer_pct: int
    confidence_threshold: float = CONFIDENCE_THRESHOLD

    # Gölge Filtreleme
    remove_shadows: bool = REMOVE_SHADOWS
    brightness_threshold: int = BRIGHTNESS_THRESHOLD
    extra_shadow_method: str = "Yok"
    shadow_text_prompt: str = "shadow, dark shaded area"
    shadow_threshold: float = 0.5

    # SAM3 Prompt Kaynakları
    add_box_prompt_search: bool = True
    add_text_prompt_search: bool = True
    text_prompt_value: str = "rooftop, building"

    # SAM3 Dilimleme
    use_sliced_sam: bool = True
    sam_slice_size: int = 1024
    sam_overlap_ratio: float = 0.4

    # Fine-Tune SAM3
    use_finetuned_sam3: bool = False
    finetuned_sam3_checkpoint_path: str = ""
    finetuned_sam3_prompt: str = "building"
    finetuned_sam3_confidence: float = 0.2

    # Görselleştirme & Geometri İyileştirme
    apply_simplify: bool = True
    simplify_level: float = 0.3
