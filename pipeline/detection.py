"""Asil tespit borusu: YOLO kutu tespiti (dilimli), SAM3 maske uretimi
(box/metin/fine-tune kaynaklari, dilimli), Shadow U-Net tahmini ve
run_detection_pipeline() - butun bu adimlari birlestirip sonucu
st.session_state icine yazan ana fonksiyon.

run_detection_pipeline(), sidebar degiskenlerine degil DetectionSettings
nesnesine (config.DetectionSettings) bakar - boylece bu dosya sidebar'dan
bagimsiz, tek basina import edilebilir/test edilebilir.
"""
import math
import os

import albumentations as A
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import box

from .config import (
    MASK_MERGE_OVERLAP_THRESHOLD,
    MAX_POLYGON_AREA_M2,
    MIN_POLYGON_AREA_M2,
    SAM3_MIN_MASK_PX,
    SHADOW_CHECKPOINT,
)
from .mask_utils import (
    apply_nms,
    bboxes_touch_near_seam,
    compute_tile_seam_lines,
    crop_mask_to_bbox,
    fast_overlap_ratio,
    is_narrow_sliver,
    mask_bbox_and_area,
    merge_seam_polygons,
)
from .models import load_finetuned_sam3, load_shadow_model
from polygon_regularizer import regularize_instance_mask_ex


def predict_yolo_sliced(model, img_rgb, conf=0.5, slice_size=640, overlap_ratio=0.25):
    h, w = img_rgb.shape[:2]
    step = int(slice_size * (1 - overlap_ratio))

    all_boxes = []
    all_scores = []

    if h <= slice_size and w <= slice_size:
        res = model.predict(source=img_rgb, conf=conf, verbose=False)[0]
        return res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy().tolist()

    y_starts = list(range(0, h, step))
    x_starts = list(range(0, w, step))

    for y in y_starts:
        for x in x_starts:
            y1 = min(y, max(0, h - slice_size))
            x1 = min(x, max(0, w - slice_size))
            y2 = min(y1 + slice_size, h)
            x2 = min(x1 + slice_size, w)

            crop = img_rgb[y1:y2, x1:x2]
            res = model.predict(source=crop, conf=conf, verbose=False)[0]

            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            scores = res.boxes.conf.cpu().numpy().tolist()

            for b, s in zip(boxes, scores):
                bx1, by1, bx2, by2 = b
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(s)

    return all_boxes, all_scores


def generate_sam_masks_sliced(sam, img_rgb, boxes, full_transform, work_dir,
                               slice_size=1024, overlap_ratio=0.2, log_fn=None):
    h, w = img_rgb.shape[:2]
    step = max(1, int(slice_size * (1 - overlap_ratio)))

    global_labeled = np.zeros((h, w), dtype=np.uint8 if False else np.uint32)
    next_label_id = 1
    n_slices_used = 0

    if h <= slice_size and w <= slice_size:
        y_starts, x_starts = [0], [0]
    else:
        y_starts = sorted(set(list(range(0, max(h - slice_size, 0) + 1, step)) + [max(h - slice_size, 0)]))
        x_starts = sorted(set(list(range(0, max(w - slice_size, 0) + 1, step)) + [max(w - slice_size, 0)]))

    slice_idx = 0
    for y1_nominal in y_starts:
        for x1_nominal in x_starts:
            y2_nominal = min(y1_nominal + slice_size, h)
            x2_nominal = min(x1_nominal + slice_size, w)

            assigned_boxes = []
            for bx1, by1, bx2, by2 in boxes:
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if x1_nominal <= cx < x2_nominal and y1_nominal <= cy < y2_nominal:
                    assigned_boxes.append((bx1, by1, bx2, by2))

            if not assigned_boxes:
                slice_idx += 1
                continue

            max_expand_px = slice_size // 2
            y1 = max(min([y1_nominal] + [int(math.floor(by1)) for _, by1, _, _ in assigned_boxes]),
                     y1_nominal - max_expand_px)
            x1 = max(min([x1_nominal] + [int(math.floor(bx1)) for bx1, _, _, _ in assigned_boxes]),
                     x1_nominal - max_expand_px)
            y2 = min(max([y2_nominal] + [int(math.ceil(by2)) for _, _, _, by2 in assigned_boxes]),
                     y2_nominal + max_expand_px)
            x2 = min(max([x2_nominal] + [int(math.ceil(bx2)) for bx1, _, bx2, _ in assigned_boxes]),
                     x2_nominal + max_expand_px)
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(h, y2), min(w, x2)

            local_boxes = []
            for bx1, by1, bx2, by2 in assigned_boxes:
                lbx1 = max(0.0, bx1 - x1)
                lby1 = max(0.0, by1 - y1)
                lbx2 = min(float(x2 - x1), bx2 - x1)
                lby2 = min(float(y2 - y1), by2 - y1)
                if lbx2 > lbx1 and lby2 > lby1:
                    local_boxes.append([lbx1, lby1, lbx2, lby2])

            if not local_boxes:
                slice_idx += 1
                continue

            crop = img_rgb[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            slice_transform = full_transform * rasterio.Affine.translation(x1, y1)

            crop_tif_path = os.path.join(work_dir, f"sam_slice_{slice_idx}.tif")
            with rasterio.open(
                crop_tif_path, "w", driver="GTiff",
                height=crop_h, width=crop_w, count=3,
                dtype=crop.dtype, crs="EPSG:3857", transform=slice_transform,
            ) as dst:
                for band in range(3):
                    dst.write(crop[:, :, band], band + 1)

            try:
                sam.set_image(crop_tif_path)
                slice_mask_path = os.path.join(work_dir, f"sam_slice_mask_{slice_idx}.png")
                try:
                    sam.generate_masks_by_boxes_inst(
                        boxes=local_boxes, box_crs=None, output=slice_mask_path,
                        dtype="uint16", multimask_output=True, unique=True,
                        min_size=SAM3_MIN_MASK_PX,
                    )
                except TypeError:
                    sam.generate_masks_by_boxes_inst(
                        boxes=local_boxes, box_crs=None, multimask_output=True,
                        min_size=SAM3_MIN_MASK_PX,
                    )
                    sam.save_masks(slice_mask_path, unique=True)

                if not os.path.exists(slice_mask_path):
                    slice_idx += 1
                    continue

                slice_labeled = cv2.imread(slice_mask_path, cv2.IMREAD_UNCHANGED)
                if slice_labeled is None:
                    slice_idx += 1
                    continue

                if slice_labeled.ndim == 3:
                    slice_labeled = slice_labeled[..., 0]
                slice_labels = [lbl for lbl in np.unique(slice_labeled) if lbl != 0]

                if slice_labels:
                    for lbl in slice_labels:
                        region = (slice_labeled == lbl)
                        global_labeled[y1:y2, x1:x2][region] = next_label_id
                        next_label_id += 1
                    n_slices_used += 1
            except Exception as e:
                if "No masks found" not in str(e) and log_fn:
                    log_fn(f"Dilim {slice_idx} ({x1},{y1})-({x2},{y2}) SAM3 hatası: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            slice_idx += 1

    return global_labeled, (h, w), n_slices_used


def generate_finetuned_sam3_masks_sliced(model, processor, img_rgb, prompt, confidence_threshold,
                                          slice_size=1024, overlap_ratio=0.2, log_fn=None):
    """generate_sam_masks_sliced ile aynı dilimleme mantığı, ama samgeo yerine
    fine-tune edilmiş transformers Sam3Model/Sam3Processor kullanır.
    Dönen: (global_labeled, (h, w), n_slices_used) - generate_sam_masks_sliced ile aynı format."""
    h, w = img_rgb.shape[:2]
    step = max(1, int(slice_size * (1 - overlap_ratio)))
    device = next(model.parameters()).device

    global_labeled = np.zeros((h, w), dtype=np.uint32)
    next_label_id = 1
    n_slices_used = 0

    if h <= slice_size and w <= slice_size:
        y_starts, x_starts = [0], [0]
    else:
        y_starts = sorted(set(list(range(0, max(h - slice_size, 0) + 1, step)) + [max(h - slice_size, 0)]))
        x_starts = sorted(set(list(range(0, max(w - slice_size, 0) + 1, step)) + [max(w - slice_size, 0)]))

    for y1 in y_starts:
        for x1 in x_starts:
            y2 = min(y1 + slice_size, h)
            x2 = min(x1 + slice_size, w)
            crop = img_rgb[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)

            try:
                with torch.no_grad():
                    inputs = processor(images=[crop_pil], text=[prompt], return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    results = processor.post_process_instance_segmentation(
                        outputs, target_sizes=[crop_pil.size[::-1]], threshold=confidence_threshold,
                    )[0]

                if "masks" in results and len(results["masks"]) > 0:
                    for m in results["masks"]:
                        m_np = m.cpu().numpy() > 0
                        if m_np.ndim > 2:
                            m_np = m_np.squeeze()
                        if m_np.sum() < SAM3_MIN_MASK_PX:
                            continue
                        global_labeled[y1:y2, x1:x2][m_np] = next_label_id
                        next_label_id += 1
                    n_slices_used += 1
            except Exception as e:
                if log_fn:
                    log_fn(f"Fine-tune SAM3 dilim ({x1},{y1})-({x2},{y2}) hatası: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return global_labeled, (h, w), n_slices_used


def predict_shadow_mask(model, device, image_rgb, image_size=512, threshold=0.5):
    orig_h, orig_w = image_rgb.shape[:2]
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).float().to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).float()

    pred_mask = probs.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h))
    return (pred_mask > threshold).astype(np.uint8)



def run_detection_pipeline(tif_path, work_dir, orig_minx, orig_miny, orig_maxx, orig_maxy, settings, yolo_model, sam):
    """Mevcut YOLO+SAM3+U-Net maskeleme/gorsellestirme borusu.
    Hem canli harita indirmesinden hem de yuklenen TIF dosyasindan
    kirpilan AOI goruntusu icin aynen kullanilir; sonuclar st.session_state
    icine yazilir (result_gdf, result_mask_raw, vb.).
    """
    debug_img = cv2.imread(tif_path)
    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

    boxes_meta = []
    boxes = []
    n_yolo_boxes = 0

    if settings.add_box_prompt_search:
        with st.spinner("YOLO bina kutularını tespit ediyor (Dilimleme Modu)..."):
            try:
                yolo_boxes_raw, yolo_scores_raw = predict_yolo_sliced(
                    yolo_model, debug_img_rgb, conf=settings.confidence_threshold, slice_size=640, overlap_ratio=0.25
                )
            except Exception as e:
                st.error(f"YOLO tespiti hatası: {e}")
                st.stop()

        boxes_meta = [(b, s, "yolo") for b, s in zip(yolo_boxes_raw, yolo_scores_raw)]
        if not boxes_meta and not settings.add_text_prompt_search:
            st.warning("Hiçbir bina tespit edilemedi.")
            st.stop()

        boxes_meta_filtered = apply_nms(boxes_meta, iou_threshold=settings.nms_threshold)

        # 1. En-Boy Oranı (Aspect Ratio) Filtresi: Aşırı uzun şeritleri (tren rayı, koridor park, yol) ele
        candidate_boxes = [item[0] for item in boxes_meta_filtered]
        boxes = []
        for bx in candidate_boxes:
            bx1, by1, bx2, by2 = bx
            w = max(1.0, bx2 - bx1)
            h = max(1.0, by2 - by1)
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio < 4.5:
                boxes.append(bx)

        n_yolo_boxes = sum(1 for item in boxes_meta_filtered if item[2] == "yolo" and item[0] in boxes)
    elif not settings.add_text_prompt_search and not settings.use_finetuned_sam3:
        st.warning("⚠️ Box tabanlı, metin tabanlı ve fine-tune SAM3 kaynaklarının hepsi kapalı - "
                   "hiçbir kaynak seçili değil.")
        st.stop()

    st.session_state["debug_img_rgb"] = debug_img_rgb
    st.session_state["debug_boxes"] = boxes
    st.session_state["debug_boxes_yolo_count"] = n_yolo_boxes

    # Parlaklık eşiği her tespitte otomatik uygulanır (kapatılamaz).
    hsv = cv2.cvtColor(debug_img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    shadow_mask_bin = (v_channel < settings.brightness_threshold).astype(np.uint8)
    st.sidebar.info(
        f"🌑 Basit parlaklık eşiği: alanın %{100 * shadow_mask_bin.mean():.1f}'i gölge olarak işaretlendi."
    )

    if settings.extra_shadow_method == "Shadow U-Net (eğitilmiş model)":
        if not os.path.exists(SHADOW_CHECKPOINT):
            st.sidebar.error(f"⚠️ Shadow U-Net checkpoint bulunamadı: `{SHADOW_CHECKPOINT}`")
        else:
            with st.spinner("Shadow U-Net gölgeleri çıkarıyor..."):
                try:
                    torch.cuda.empty_cache()
                    shadow_model, shadow_device = load_shadow_model(SHADOW_CHECKPOINT, encoder="resnet34")
                    unet_shadow_mask = predict_shadow_mask(shadow_model, shadow_device, debug_img_rgb, threshold=settings.shadow_threshold)
                    shadow_mask_bin = shadow_mask_bin | unet_shadow_mask
                    st.sidebar.info(
                        f"Shadow U-Net eklendi: birleşim alanın %{100 * shadow_mask_bin.mean():.1f}'i gölge oldu."
                    )
                except Exception as e:
                    st.warning(f"Shadow U-Net hatası: {e}")

    with st.spinner("SAM3 maskeleri üretiyor ve geometriler optimize ediliyor..."):
        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs = src.crs

        sam.set_image(tif_path)

        st.session_state["debug_shadow_state"] = {
            "remove_shadows": settings.remove_shadows,
            "extra_shadow_method": settings.extra_shadow_method,
            "shadow_text_prompt": settings.shadow_text_prompt,
        }

        if settings.extra_shadow_method == "SAM3 (zero-shot metin prompt)" and settings.shadow_text_prompt.strip():
            with st.spinner(f"SAM3 zero-shot gölge tespiti: '{settings.shadow_text_prompt}'..."):
                try:
                    aoi_mask_shadow_path = os.path.join(work_dir, "aoi_mask_shadow.png")
                    sam.generate_masks(settings.shadow_text_prompt.strip())
                    sam.save_masks(aoi_mask_shadow_path, unique=True)

                    shadow_labeled = cv2.imread(aoi_mask_shadow_path, cv2.IMREAD_UNCHANGED)
                    if shadow_labeled is not None:
                        if shadow_labeled.ndim == 3:
                            shadow_labeled = shadow_labeled[..., 0]
                        sam3_shadow_mask = (shadow_labeled > 0).astype(np.uint8)
                        shadow_mask_bin = shadow_mask_bin | sam3_shadow_mask
                        st.sidebar.info(
                            f"🌑 SAM3 gölge tespiti eklendi: birleşim alanın "
                            f"%{100 * shadow_mask_bin.mean():.1f}'i gölge oldu."
                        )
                        st.session_state["debug_shadow_state"]["sam3_result"] = (
                            f"başarılı, %{100 * sam3_shadow_mask.mean():.1f} gölge, "
                            f"{len(np.unique(shadow_labeled)) - 1} etiket bulundu"
                        )
                    else:
                        st.sidebar.warning("SAM3 gölge maskesi dosyası okunamadı.")
                        st.session_state["debug_shadow_state"]["sam3_result"] = "dosya okunamadı (None döndü)"
                except Exception as e:
                    if "No masks found" in str(e):
                        st.sidebar.warning(
                            f"🌑 SAM3, '{settings.shadow_text_prompt}' promptuyla hiç gölge bulamadı (0 tespit)."
                        )
                        st.session_state["debug_shadow_state"]["sam3_result"] = "0 tespit (SAM3 gölgeyi tanımadı)"
                    else:
                        st.sidebar.warning(f"SAM3 gölge tespiti hatası: {e}")
                        st.session_state["debug_shadow_state"]["sam3_result"] = f"HATA: {e}"
        else:
            st.session_state["debug_shadow_state"]["sam3_result"] = (
                "blok çalışmadı (extra_shadow_method != SAM3)"
            )

        img_shape_ref = debug_img_rgb.shape[:2]

        unique_labels = []
        aoi_mask_labeled = None

        if settings.add_box_prompt_search and boxes:
            if settings.use_sliced_sam:
                with st.spinner(
                    f"SAM3 dilimleyerek çalıştırılıyor "
                    f"(dilim={settings.sam_slice_size}px, örtüşme=%{int(settings.sam_overlap_ratio * 100)})..."
                ):
                    try:
                        aoi_mask_labeled, img_shape_ref, n_slices_used = generate_sam_masks_sliced(
                            sam, debug_img_rgb, boxes, transform, work_dir,
                            slice_size=settings.sam_slice_size, overlap_ratio=settings.sam_overlap_ratio,
                            log_fn=st.sidebar.warning,
                        )
                    except Exception as e:
                        st.error(f"SAM3 dilimli maske üretimi sırasında hata: {e}")
                        st.stop()

                    unique_labels = [lbl for lbl in np.unique(aoi_mask_labeled) if lbl != 0]
                    st.sidebar.info(
                        f"SAM3 dilimleme: {n_slices_used} dilimde maske üretildi, "
                        f"toplam {len(unique_labels)} instance bulundu."
                    )
                    if not unique_labels and not settings.add_text_prompt_search:
                        st.warning("SAM3 maske üretemedi (dilimli mod).")
                        st.stop()
            else:
                aoi_mask_path = os.path.join(work_dir, "aoi_mask.png")
                try:
                    try:
                        sam.generate_masks_by_boxes_inst(
                            boxes=boxes, box_crs=None, output=aoi_mask_path,
                            dtype="uint16", multimask_output=True, unique=True,
                            min_size=SAM3_MIN_MASK_PX,
                        )
                    except TypeError:
                        sam.generate_masks_by_boxes_inst(
                            boxes=boxes, box_crs=None, multimask_output=True,
                            min_size=SAM3_MIN_MASK_PX,
                        )
                        sam.save_masks(aoi_mask_path, unique=True)
                except Exception as e:
                    st.error(f"SAM3 maske üretimi sırasında hata: {e}")
                    st.stop()

                aoi_mask_labeled = cv2.imread(aoi_mask_path, cv2.IMREAD_UNCHANGED)
                if aoi_mask_labeled is None:
                    st.error("SAM3 maske dosyası okunamadı.")
                    st.stop()
                if aoi_mask_labeled.ndim == 3:
                    aoi_mask_labeled = aoi_mask_labeled[..., 0]
                unique_labels = [lbl for lbl in np.unique(aoi_mask_labeled) if lbl != 0]
                if not unique_labels and not settings.add_text_prompt_search:
                    st.warning("SAM3 maske üretemedi.")
                    st.stop()
                img_shape_ref = aoi_mask_labeled.shape

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        polygon_list = []
        polygon_regularized_flags = []
        # MIN_POLYGON_AREA_M2, MAX_POLYGON_AREA_M2, MASK_MERGE_OVERLAP_THRESHOLD artik config.py'den import ediliyor.

        y_seams, x_seams = compute_tile_seam_lines(
            img_shape_ref[0], img_shape_ref[1], settings.sam_slice_size, settings.sam_overlap_ratio
        ) if settings.use_sliced_sam else ([], [])

        mask_before_shadow = np.zeros(img_shape_ref, dtype=np.uint8)
        aoi_mask_bin = np.zeros(img_shape_ref, dtype=np.uint8)

        if shadow_mask_bin is not None:
            _shadow_bbox, _shadow_area = mask_bbox_and_area(shadow_mask_bin)
            if _shadow_bbox is None:
                shadow_mask_bin = np.zeros_like(shadow_mask_bin)
            else:
                sy1, sy2, sx1, sx2 = _shadow_bbox
                shadow_cropped = crop_mask_to_bbox(shadow_mask_bin, _shadow_bbox)
                shadow_local_transform = transform * rasterio.Affine.translation(sx1, sy1)

                shadow_union, _ = regularize_instance_mask_ex(
                    shadow_cropped, shadow_local_transform, min_area_m2=MIN_POLYGON_AREA_M2)

                new_shadow_mask_bin = np.zeros_like(shadow_mask_bin)
                if shadow_union is not None and not shadow_union.is_empty:
                    shadow_geoms = (
                        list(shadow_union.geoms)
                        if shadow_union.geom_type == "MultiPolygon"
                        else [shadow_union]
                    )
                    rasterized_crop = rasterize(
                        [(geom, 1) for geom in shadow_geoms],
                        out_shape=(sy2 - sy1, sx2 - sx1),
                        transform=shadow_local_transform,
                        fill=0,
                        dtype=np.uint8,
                    )
                    new_shadow_mask_bin[sy1:sy2, sx1:sx2] = rasterized_crop
                shadow_mask_bin = new_shadow_mask_bin
            st.sidebar.info(
                f"🧹 Gölge maskesi temizlendi (regularize): alanın "
                f"%{100 * shadow_mask_bin.mean():.1f}'i gölge olarak kaldı."
            )

        candidate_masks = []

        for label_id in unique_labels:
            inst_mask = (aoi_mask_labeled == label_id).astype(np.uint8)

            # --- MASKELEME AŞAMASI İYİLEŞTİRMELERİ ---
            # 1. En Büyük Bağlantılı Bileşeni Al (Parazit kopuk parçacıkları ve saçakları eler)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inst_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                inst_mask = (labels == largest_label).astype(np.uint8)

            # 2. Renk / Parlaklık Eşiklemesi ile Düşey Cephe Duvarlarını Ayıkla
            if np.any(inst_mask):
                gray_img = cv2.cvtColor(debug_img_rgb, cv2.COLOR_RGB2GRAY)
                mask_gray = np.where(inst_mask == 1, gray_img, 0)
                fg_pixels = gray_img[inst_mask == 1]
                if len(fg_pixels) > 0 and np.std(fg_pixels) > 15:
                    _, binary_roof = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    otsu_mask = (binary_roof > 0).astype(np.uint8)
                    # Aşırı parça kaybını önlemek için alan kontrolü
                    if otsu_mask.sum() >= 0.45 * inst_mask.sum():
                        inst_mask = otsu_mask

            inst_mask_255 = (inst_mask * 255).astype(np.uint8)
            inst_mask_255 = cv2.morphologyEx(inst_mask_255, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            inst_mask_255 = cv2.morphologyEx(inst_mask_255, cv2.MORPH_OPEN, open_kernel, iterations=1)

            blurred = cv2.GaussianBlur(inst_mask_255, (7, 7), sigmaX=1.5)
            _, inst_mask_255 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            inst_mask = (inst_mask_255 > 0).astype(np.uint8)
            mask_before_shadow |= inst_mask

            if shadow_mask_bin is not None:
                inst_mask = inst_mask & (~shadow_mask_bin.astype(bool)).astype(np.uint8)
                inst_mask_255_post = (inst_mask * 255).astype(np.uint8)
                inst_mask_255_post = cv2.morphologyEx(
                    inst_mask_255_post, cv2.MORPH_CLOSE, close_kernel, iterations=1
                )
                inst_mask = (inst_mask_255_post > 0).astype(np.uint8)

            if inst_mask.sum() == 0:
                continue

            _bbox, _area = mask_bbox_and_area(inst_mask)
            candidate_masks.append((crop_mask_to_bbox(inst_mask, _bbox), _area, _bbox))

        if settings.add_text_prompt_search and settings.text_prompt_value.strip():
            with st.spinner(f"SAM3 metin prompt ile ek tarama yapılıyor: '{settings.text_prompt_value}'..."):
                try:
                    if settings.use_sliced_sam:
                        aoi_mask_text_path = os.path.join(work_dir, "aoi_mask_text.tif")
                        overlap_px = int(settings.sam_slice_size * settings.sam_overlap_ratio)
                        try:
                            sam.generate_masks_tiled(
                                source=tif_path,
                                prompt=settings.text_prompt_value.strip(),
                                output=aoi_mask_text_path,
                                tile_size=settings.sam_slice_size,
                                overlap=overlap_px,
                                min_size=SAM3_MIN_MASK_PX,
                                unique=True,
                                dtype="uint16",
                            )
                        except TypeError:
                            sam.generate_masks_tiled(
                                source=tif_path,
                                prompt=settings.text_prompt_value.strip(),
                                output=aoi_mask_text_path,
                                tile_size=settings.sam_slice_size,
                                overlap=overlap_px,
                            )

                        try:
                            with rasterio.open(aoi_mask_text_path) as _src_text:
                                aoi_mask_text_labeled = _src_text.read(1)
                        except Exception:
                            aoi_mask_text_labeled = cv2.imread(aoi_mask_text_path, cv2.IMREAD_UNCHANGED)
                        if aoi_mask_text_labeled is None:
                            st.sidebar.warning(
                                "⚠️ generate_masks_tiled çıktı dosyası (aoi_mask_text.tif) "
                                "hem rasterio hem cv2 ile okunamadı - maske kayboldu."
                            )
                        st.sidebar.info(
                            f"generate_masks_tiled: dilim={settings.sam_slice_size}px, "
                            f"örtüşme={overlap_px}px ile tarandı."
                        )
                    else:
                        aoi_mask_text_path = os.path.join(work_dir, "aoi_mask_text.png")
                        sam.generate_masks(settings.text_prompt_value.strip(), min_size=SAM3_MIN_MASK_PX)
                        sam.save_masks(aoi_mask_text_path, unique=True)
                        aoi_mask_text_labeled = cv2.imread(aoi_mask_text_path, cv2.IMREAD_UNCHANGED)

                    if aoi_mask_text_labeled is not None:
                        if aoi_mask_text_labeled.ndim == 3:
                            aoi_mask_text_labeled = aoi_mask_text_labeled[..., 0]
                        text_labels = [lbl for lbl in np.unique(aoi_mask_text_labeled) if lbl != 0]

                        n_text_masks = 0
                        for label_id in text_labels:
                            t_mask = (aoi_mask_text_labeled == label_id).astype(np.uint8)

                            # --- METİN ARAMASI İÇİN MASKELEME FİLTRELERİ ---
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(t_mask, connectivity=8)
                            if num_labels > 1:
                                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                                t_mask = (labels == largest_label).astype(np.uint8)

                            if np.any(t_mask):
                                gray_img = cv2.cvtColor(debug_img_rgb, cv2.COLOR_RGB2GRAY)
                                mask_gray = np.where(t_mask == 1, gray_img, 0)
                                fg_pixels = gray_img[t_mask == 1]
                                if len(fg_pixels) > 0 and np.std(fg_pixels) > 15:
                                    _, binary_roof = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    otsu_mask = (binary_roof > 0).astype(np.uint8)
                                    if otsu_mask.sum() >= 0.45 * t_mask.sum():
                                        t_mask = otsu_mask

                            t_mask_255 = (t_mask * 255).astype(np.uint8)
                            t_mask_255 = cv2.morphologyEx(t_mask_255, cv2.MORPH_CLOSE, close_kernel, iterations=1)
                            t_mask_255 = cv2.morphologyEx(t_mask_255, cv2.MORPH_OPEN, open_kernel, iterations=1)
                            blurred_t = cv2.GaussianBlur(t_mask_255, (7, 7), sigmaX=1.5)
                            _, t_mask_255 = cv2.threshold(blurred_t, 127, 255, cv2.THRESH_BINARY)
                            t_mask = (t_mask_255 > 0).astype(np.uint8)

                            mask_before_shadow |= t_mask

                            if shadow_mask_bin is not None:
                                t_mask = t_mask & (~shadow_mask_bin.astype(bool)).astype(np.uint8)

                            if t_mask.sum() == 0:
                                continue

                            t_bbox, t_area = mask_bbox_and_area(t_mask)
                            t_mask_cropped = crop_mask_to_bbox(t_mask, t_bbox)

                            max_overlap_with_existing = 0.0
                            for existing_mask, existing_area, existing_bbox in candidate_masks:
                                ratio = fast_overlap_ratio(
                                    t_mask_cropped, t_bbox, t_area,
                                    existing_mask, existing_bbox, existing_area,
                                )
                                if ratio > max_overlap_with_existing:
                                    max_overlap_with_existing = ratio
                                    if max_overlap_with_existing >= MASK_MERGE_OVERLAP_THRESHOLD:
                                        break

                            if max_overlap_with_existing >= MASK_MERGE_OVERLAP_THRESHOLD:
                                continue

                            candidate_masks.append((t_mask_cropped, t_area, t_bbox))
                            n_text_masks += 1

                        if n_text_masks > 0:
                            st.sidebar.info(f"🔤 Metin prompt ('{settings.text_prompt_value}') ile {n_text_masks} ek maske bulundu.")
                except Exception as e:
                    st.sidebar.warning(f"SAM3 metin prompt taraması hatası: {e}")

        if settings.use_finetuned_sam3 and os.path.exists(settings.finetuned_sam3_checkpoint_path):
            with st.spinner(f"Fine-tune SAM3 modeli ek tarama yapıyor: '{settings.finetuned_sam3_prompt}'..."):
                try:
                    ft_model, ft_processor, ft_device = load_finetuned_sam3(settings.finetuned_sam3_checkpoint_path)

                    ft_labeled, ft_shape, ft_n_slices = generate_finetuned_sam3_masks_sliced(
                        ft_model, ft_processor, debug_img_rgb,
                        prompt=settings.finetuned_sam3_prompt, confidence_threshold=settings.finetuned_sam3_confidence,
                        slice_size=settings.sam_slice_size, overlap_ratio=settings.sam_overlap_ratio,
                        log_fn=st.sidebar.warning,
                    )
                    ft_labels = [lbl for lbl in np.unique(ft_labeled) if lbl != 0]

                    n_ft_masks = 0
                    for label_id in ft_labels:
                        ft_mask = (ft_labeled == label_id).astype(np.uint8)

                        # --- Aynı maskeleme filtreleri (box/text kaynaklarıyla birebir aynı) ---
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ft_mask, connectivity=8)
                        if num_labels > 1:
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            ft_mask = (labels == largest_label).astype(np.uint8)

                        if np.any(ft_mask):
                            gray_img = cv2.cvtColor(debug_img_rgb, cv2.COLOR_RGB2GRAY)
                            mask_gray = np.where(ft_mask == 1, gray_img, 0)
                            fg_pixels = gray_img[ft_mask == 1]
                            if len(fg_pixels) > 0 and np.std(fg_pixels) > 15:
                                _, binary_roof = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                otsu_mask = (binary_roof > 0).astype(np.uint8)
                                if otsu_mask.sum() >= 0.45 * ft_mask.sum():
                                    ft_mask = otsu_mask

                        ft_mask_255 = (ft_mask * 255).astype(np.uint8)
                        ft_mask_255 = cv2.morphologyEx(ft_mask_255, cv2.MORPH_CLOSE, close_kernel, iterations=1)
                        ft_mask_255 = cv2.morphologyEx(ft_mask_255, cv2.MORPH_OPEN, open_kernel, iterations=1)
                        blurred_ft = cv2.GaussianBlur(ft_mask_255, (7, 7), sigmaX=1.5)
                        _, ft_mask_255 = cv2.threshold(blurred_ft, 127, 255, cv2.THRESH_BINARY)
                        ft_mask = (ft_mask_255 > 0).astype(np.uint8)

                        mask_before_shadow |= ft_mask

                        if shadow_mask_bin is not None:
                            ft_mask = ft_mask & (~shadow_mask_bin.astype(bool)).astype(np.uint8)

                        if ft_mask.sum() == 0:
                            continue

                        ft_bbox, ft_area = mask_bbox_and_area(ft_mask)
                        ft_mask_cropped = crop_mask_to_bbox(ft_mask, ft_bbox)

                        max_overlap_with_existing = 0.0
                        for existing_mask, existing_area, existing_bbox in candidate_masks:
                            ratio = fast_overlap_ratio(
                                ft_mask_cropped, ft_bbox, ft_area,
                                existing_mask, existing_bbox, existing_area,
                            )
                            if ratio > max_overlap_with_existing:
                                max_overlap_with_existing = ratio
                                if max_overlap_with_existing >= MASK_MERGE_OVERLAP_THRESHOLD:
                                    break

                        if max_overlap_with_existing >= MASK_MERGE_OVERLAP_THRESHOLD:
                            continue

                        candidate_masks.append((ft_mask_cropped, ft_area, ft_bbox))
                        n_ft_masks += 1

                    if n_ft_masks > 0:
                        st.sidebar.info(
                            f"🧠 Fine-tune SAM3 ile {n_ft_masks} ek maske bulundu "
                            f"({ft_n_slices} dilimde tespit yapıldı)."
                        )
                except Exception as e:
                    st.sidebar.warning(f"Fine-tune SAM3 tarama hatası: {e}")

        candidate_masks.sort(key=lambda x: x[1], reverse=True)
        kept_masks = []
        n_merged_by_seam = 0
        for inst_mask, area, bbox in candidate_masks:
            merged_into_existing = False
            for idx, (kept_mask, kept_area, kept_bbox) in enumerate(kept_masks):
                ratio = fast_overlap_ratio(inst_mask, bbox, area, kept_mask, kept_bbox, kept_area)
                should_merge = ratio >= MASK_MERGE_OVERLAP_THRESHOLD
                if not should_merge and (y_seams or x_seams):
                    if bboxes_touch_near_seam(bbox, kept_bbox, y_seams, x_seams):
                        should_merge = True
                        n_merged_by_seam += 1
                if should_merge:
                    uy1 = min(bbox[0], kept_bbox[0])
                    uy2 = max(bbox[1], kept_bbox[1])
                    ux1 = min(bbox[2], kept_bbox[2])
                    ux2 = max(bbox[3], kept_bbox[3])
                    merged_mask = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.uint8)
                    merged_mask[bbox[0] - uy1:bbox[1] - uy1, bbox[2] - ux1:bbox[3] - ux1] |= inst_mask
                    merged_mask[kept_bbox[0] - uy1:kept_bbox[1] - uy1, kept_bbox[2] - ux1:kept_bbox[3] - ux1] |= kept_mask
                    merged_bbox = (uy1, uy2, ux1, ux2)
                    merged_area = int(merged_mask.sum())
                    kept_masks[idx] = (merged_mask, merged_area, merged_bbox)
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                kept_masks.append((inst_mask, area, bbox))

        if n_merged_by_seam > 0:
            st.sidebar.info(f"🧵 {n_merged_by_seam} dilim-sınırı bitişikliğiyle (seam) birleştirildi.")

        n_merged = len(candidate_masks) - len(kept_masks)
        if n_merged > 0:
            st.sidebar.info(f"🔗 {n_merged} örtüşen (çift) maske birleştirildi.")

        for inst_mask, _, bbox in kept_masks:
            by1, by2, bx1, bx2 = bbox
            aoi_mask_bin[by1:by2, bx1:bx2] |= inst_mask

            local_transform = transform * rasterio.Affine.translation(bx1, by1)
            inst_union, was_regularized = regularize_instance_mask_ex(
                inst_mask, local_transform, min_area_m2=MIN_POLYGON_AREA_M2)

            if inst_union is None or inst_union.is_empty:
                continue

            if inst_union.geom_type == "MultiPolygon":
                for g in inst_union.geoms:
                    # 2. Alan Sınırı: Minimum ve Maksimum Alan Kontrolü
                    if MIN_POLYGON_AREA_M2 <= g.area <= MAX_POLYGON_AREA_M2:
                        polygon_list.append(g)
                        polygon_regularized_flags.append(False)
            elif inst_union.geom_type == "Polygon" and not inst_union.is_empty:
                # 2. Alan Sınırı: Minimum ve Maksimum Alan Kontrolü
                if MIN_POLYGON_AREA_M2 <= inst_union.area <= MAX_POLYGON_AREA_M2:
                    polygon_list.append(inst_union)
                    polygon_regularized_flags.append(was_regularized)

        if polygon_list:
            polygon_list = merge_seam_polygons(polygon_list, seam_tol_m=1.0)
            polygon_list.sort(key=lambda p: p.area, reverse=True)
            final_resolved_polys = []
            final_flags = []

            for poly in polygon_list:
                current = poly
                for existing in final_resolved_polys:
                    if current.intersects(existing):
                        current = current.difference(existing)

                if not current.is_empty and current.area >= MIN_POLYGON_AREA_M2:
                    sub_geoms = current.geoms if current.geom_type == "MultiPolygon" else [current]
                    for sub_p in sub_geoms:
                        if MIN_POLYGON_AREA_M2 <= sub_p.area <= MAX_POLYGON_AREA_M2 and not is_narrow_sliver(sub_p, min_width_m=1.5):
                            final_resolved_polys.append(sub_p)
                            final_flags.append(False)

            polygon_list = final_resolved_polys
            polygon_regularized_flags = final_flags

        removed_ratio = None
        if shadow_mask_bin is not None:
            pre_count = int(mask_before_shadow.sum())
            post_count = int(aoi_mask_bin.sum())
            removed_ratio = (pre_count - post_count) / pre_count if pre_count > 0 else 0.0

        if not polygon_list:
            st.warning("SAM geçerli bir poligon üretemedi.")
            st.stop()

        gdf_proj = gpd.GeoDataFrame(geometry=polygon_list, crs=crs)
        gdf_proj["_regularized"] = polygon_regularized_flags
        _n_reg = sum(polygon_regularized_flags)
        _n_total = len(polygon_regularized_flags)
        st.sidebar.info(
            f"📐 Regularize: {_n_reg}/{_n_total} bina dik-acili hale getirildi "
            f"({_n_total - _n_reg} tanesi guard tarafindan reddedilip smooth_vector'a dustu)."
        )

        if len(gdf_proj) > 0 and settings.apply_simplify:
            needs_smooth = ~gdf_proj["_regularized"]
            if needs_smooth.any():
                try:
                    from samgeo.common import smooth_vector

                    to_smooth = gdf_proj[needs_smooth].copy()
                    n_before_smooth = len(to_smooth)

                    to_smooth = smooth_vector(
                        to_smooth,
                        smooth_iterations=3,
                        preserve_area=True,
                        area_tolerance=1.5,
                    )

                    to_smooth["geometry"] = to_smooth["geometry"].simplify(
                        tolerance=settings.simplify_level, preserve_topology=True
                    )

                    common_idx = to_smooth.index.intersection(gdf_proj.index)
                    gdf_proj.loc[common_idx, "geometry"] = to_smooth.loc[common_idx, "geometry"]

                    n_dropped = n_before_smooth - len(common_idx)
                    if n_dropped > 0:
                        st.sidebar.info(
                            f"ℹ️ Yumuşatma sırasında {n_dropped} geometri düşürüldü "
                            f"(muhtemelen geçersiz/boş çıktı), ham hâliyle bırakıldı."
                        )
                except Exception as e:
                    st.sidebar.warning(f"Geometri yumuşatma hatası: {e}")

        gdf_proj = gdf_proj.drop(columns=["_regularized"])

        gdf_final = gdf_proj.to_crs("EPSG:4326")

        orig_geom = box(orig_minx, orig_miny, orig_maxx, orig_maxy)
        gdf_final = gdf_final[gdf_final.geometry.intersects(orig_geom)].reset_index(drop=True)

        st.session_state["result_gdf"] = gdf_final
        st.session_state["result_mask_raw"] = aoi_mask_bin
        st.session_state["result_mask_pre_shadow"] = mask_before_shadow
        st.session_state["result_shadow_mask"] = shadow_mask_bin
        st.session_state["shadow_removed_ratio"] = removed_ratio
        st.session_state["result_orig_box"] = orig_geom
        st.session_state["result_transform"] = transform
        st.session_state["result_crs"] = crs


