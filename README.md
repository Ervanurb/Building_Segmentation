# Building Segmentation — Uydu Görüntülerinden Bina Çatısı Tespiti

Uydu ve hava görüntülerinden bina çatılarını tespit edip coğrafi referanslı poligonlara dönüştüren bir boru hattı. **YOLO11m** bina kutularını bulur, **SAM3** bu kutulardan (ve metin promptlarından) piksel düzeyinde maske üretir, gölge filtreleme ve geometri düzenleme adımları maskeleri temizler. Sonuçlar **Streamlit** tabanlı etkileşimli bir harita panelinde gösterilir.

Panelde iki çalışma modu var:

- **Canlı harita:** Haritada bir bölgeye yakınlaşıp dikdörtgen aracıyla alan seçilir; uydu karoları (Esri World Imagery veya Google Satellite) paralel indirilip tespit çalıştırılır.
- **TIF yükleme:** Kendi `.tif/.tiff` uydu ya da hava görüntünüzü yükleyip mini harita üzerinde alan seçersiniz; aynı boru hattı çalışır. Coğrafi referansı bozuk veya eksik dosyalar otomatik onarılmaya çalışılır.

---

## Boru Hattı

```
 Alan seçimi (harita / TIF)
          │
          ▼
 Görüntü indirme / kırpma  ──►  GeoTIFF (zoom 19, kenar payıyla genişletilmiş)
          │
          ▼
 YOLO11m (dilimli tahmin) ──►  bina kutuları + NMS
          │
          ▼
 SAM3 maske üretimi (dilimli, örtüşmeli)
   ├─ Box prompt   : YOLO kutuları
   ├─ Text prompt  : "rooftop, building" (YOLO'nun kaçırdığı binalar için)
   └─ Fine-tune SAM3 (isteğe bağlı ek kaynak)
          │
          ▼
 Maske birleştirme ve filtreleme
   (çift maske birleştirme, dilim sınırı onarımı, alan / en-boy / ince şerit filtreleri)
          │
          ▼
 Gölge filtreleme
   ├─ HSV parlaklık eşiği (her zaman açık)
   └─ Ek yöntem: Shadow U-Net  veya  SAM3 zero-shot metin prompt
          │
          ▼
 Vektörleştirme ve geometri düzenleme (simplify / regularize)
          │
          ▼
 Harita üzerinde poligonlar + ara aşama görselleri + YOLO metrikleri
```

---

## Proje Yapısı

```
Building_Segmentation/
├── app_live.py                  # Streamlit uygulamasının giriş noktası
├── pipeline/
│   ├── config.py                # Sabitler, hazır konumlar, DetectionSettings
│   ├── geo_utils.py             # Web Mercator karo matematiği, canlı görüntü indirme
│   ├── tif_upload.py            # TIF önizleme, AOI kırpma, coğrafi referans onarımı
│   ├── models.py                # YOLO, SamGeo3, Shadow U-Net, fine-tune SAM3 yükleme
│   ├── detection.py             # run_detection_pipeline(): asıl tespit akışı
│   ├── mask_utils.py            # NMS, dilim sınırı birleştirme, maske/poligon yardımcıları
│   ├── ui_sidebar.py            # Kenar çubuğu ayarları
│   ├── ui_map_section.py        # Canlı harita ile tespit bölümü
│   ├── ui_tif_section.py        # TIF ile tespit bölümü
│   └── ui_results_section.py    # Sonuçlar, ara görseller, metrikler
│
├── mask_postprocess.py          # Ortak maske temizleme + raster→GeoDataFrame
├── polygon_regularizer.py       # Çatı poligonu düzenleme, ince şerit (sliver) filtresi
│
├── egitim.py                    # YOLO11m eğitimi (Ultralytics)
├── data.yaml                    # YOLO veri seti tanımı (tek sınıf: bina)
├── convert_coco_to_yolo.py      # COCO bbox → YOLO formatı
├── yolo_to_coco.py              # YOLO bbox/poligon → COCO JSON
│
├── yolo_samgeo3_pipeline.py     # Toplu (offline) YOLO → SAM3 maskeleme
├── run_pipeline_text_prompt.py  # YOLO kutusu + SAM3 metin promptu hibrit akış
│
├── shadow_eo_dataset.py         # S-EO veri seti okuyucu (tar'dan akışla okuma)
├── train_shadow.py              # Shadow U-Net eğitimi (S-EO)
├── extract_shadows.py           # S-EO gölge maskesi arşivlerini açma
├── verify_mask_semantics.py     # S-EO maske değerlerinin anlamını doğrulama
├── analyze_predictions_seo.py   # Shadow U-Net tahmini vs. gerçek maske (IoU)
└── compare_shadow_rooftop.py    # SAM3 çatı maskesinden gölgeyi çıkarma (offline)
```

---

## Kurulum

### Gereksinimler

- Python 3.12
- CUDA destekli NVIDIA GPU 
- Windows üzerinde geliştirildi; Linux'ta da çalışması beklenir
- SAM3 ağırlıkları Hugging Face'te erişim izniyle indirilir: [`facebook/sam3`](https://huggingface.co/facebook/sam3) sayfasından erişim isteyip `huggingface-cli login` ile giriş yapın

### Bağımlılıklar

```bash
# PyTorch'u kendi CUDA sürümünüze göre kurun: https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install ultralytics segment-geospatial transformers \
            segmentation-models-pytorch albumentations schedulefree \
            streamlit streamlit-folium folium python-dotenv \
            rasterio geopandas shapely opencv-python pillow \
            numpy pandas matplotlib requests pyyaml
```

> Windows'ta SAM3 için Triton kurulumu gerekebilir (`triton-windows`).

### Ortam değişkenleri

Google Satellite katmanını kullanmak için proje kök dizininde bir `.env` dosyası oluşturun:

```env
GOOGLE_MAPS_API_KEY=anahtarınız
```

Anahtar yoksa uygulama yalnızca **Esri World Imagery** ile çalışır.

### Model ağırlıkları ve yollar

Model yolları şu an `pipeline/config.py` içinde sabit tanımlı; kendi ortamınıza göre güncelleyin:

| Değişken | Açıklama |
|---|---|
| `RUNS_DIR` | YOLO eğitim çıktıları. Uygulama `yolov11m_geosam_model*/weights/best.pt` desenine uyan **en yeni** ağırlığı otomatik seçer. |
| `SHADOW_UNET_DIR` / `SHADOW_CHECKPOINT` | Shadow U-Net checkpoint'i (`best.pt`) |
| Fine-tune SAM3 yolu | Kenar çubuğundan girilir (varsayılan: `sam3_checkpoints/best_sam3_building.pth`) |

Model ağırlıkları boyutları nedeniyle repoya dahil değildir.

---

## Kullanım

```bash
streamlit run app_live.py
```

1. Kenar çubuğundan uydu kaynağını ve tespit ayarlarını seçin.
2. Haritada bir bölgeye gidin (hazır konum kısayolları mevcut) ya da bir TIF yükleyin.
3. Dikdörtgen aracıyla alanı seçip tespiti başlatın.
4. Sonuç haritasında poligonları, **Ara Aşama Görselleri** sekmesinde YOLO kutularını ve ham maskeleri inceleyin.

### Önemli ayarlar

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| NMS IoU eşiği | 0.45 | Çakışan kutuları eler |
| Kenar payı | %20 | Seçim sınırında yarım kalan binaları tamamlamak için indirme alanını genişletir |
| Box tabanlı SAM3 | Açık | YOLO kutularını SAM3'e box prompt olarak verir |
| Metin promptuyla ek tarama | Açık, `rooftop, building` | YOLO'nun kaçırdığı binaları bulur. 1–2 terimle başlamak önerilir; çok fazla eş anlamlı terim komşu binaların birleşmesine yol açabilir |
| SAM3 dilimleme | Açık, 1024 px, %40 örtüşme | Geniş alanlarda boş/geçersiz maskeleri önler |
| Ek gölge yöntemi | Yok | Shadow U-Net veya SAM3 zero-shot; parlaklık eşiğiyle birleşimi (union) kullanılır |
| Fine-tune SAM3 | Kapalı | Açıldığında ek maske kaynağı olarak yüklenir; kapalıyken VRAM kullanmaz |
| Kenar düzleştirme | Açık, 0.3 m | Poligon tırtıklarını giderir |

Sabit eşikler (`pipeline/config.py`): YOLO güven eşiği `0.2`, poligon alanı `45–1200 m²`, indirme zoom seviyesi `19`.

---

## Model Eğitimi

### YOLO11m (bina kutuları)

```bash
python convert_coco_to_yolo.py   # COCO veri setini YOLO formatına çevirir
python egitim.py                 # data.yaml ile eğitim
```

Veri artırma: 180° döndürme, yatay/dikey çevirme, mosaic.

### Shadow U-Net (gölge segmentasyonu)

[S-EO](https://huggingface.co/datasets/emasquil/shadow-eo) veri seti üzerinde, [arXiv:2504.06920](https://arxiv.org/abs/2504.06920) makalesindeki kurulum takip edilerek eğitilir: U-Net + ResNet34 encoder + SCSE attention, AdamW Schedule-Free, ignore maskeli focal loss. Büyük görüntü arşivi diske açılmadan tar'dan akışla okunur; 8 GB VRAM için gradient accumulation kullanılır.

```bash
python extract_shadows.py          # gölge maskesi arşivlerini aç
python verify_mask_semantics.py    # maske değerlerini doğrula (eğitimden önce)
python train_shadow.py             # eğitim
python train_shadow.py --resume <run_dir>/last.pt
python analyze_predictions_seo.py  # validasyon görselleri + IoU
```
---

## Bilinen Sınırlamalar
- SAM3 zero-shot metin promptu "gölge" kavramını tek başına güvenilir biçimde bulamıyor; bu yüzden parlaklık eşiği her zaman açık tutuluyor.
- Dosya yolları Windows'a göre sabit kodlanmış; başka bir ortamda `pipeline/config.py` ve script başlarındaki yolların güncellenmesi gerekir.

---
## Proje Hakkında

Bu proje, bilgisayarlı görü ve coğrafi yapay zekâ alanında kendimi geliştirmek amacıyla yürüttüğüm bir eğitim çalışmasıdır. Süreç boyunca nesne tespiti, foundation model fine-tuning'i, gölge segmentasyonu ve coğrafi veri işleme konularında deneyler yapıldı. Repodaki bazı scriptler bu deneylerin ara adımlarıdır ve dosya yolları kendi çalışma ortamıma göre ayarlanmıştır.

