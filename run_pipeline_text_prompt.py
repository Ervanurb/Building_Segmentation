"""
YOLO + SAM3 (box + text prompt hibrit) pipeline.

Mantik:
    1. YOLO ile bina kutulari tespit edilir (KONUM bilgisi).
    2. SAM3 text-mode ile ("rooftop" prompt'u) goruntudeki TUM olasi
       cati/bina maskeleri uretilir (SEMANTIK filtre).
    3. Sadece YOLO kutulariyla yeterince ortusen (IoU >= IOU_THRESHOLD)
       SAM3 maskeleri tutulur ve tek bir binary maskede birlestirilir.

Not: SamGeo3'un generate_masks_by_boxes_inst() (instance-interactivity,
SAM1-tarzi) metodu text prompt PARAMETRESI ALMAZ; generate_masks(prompt)
ise kutu almaz. Bu yuzden iki cagriyi ayri yapip IoU ile eslestiriyoruz.
"""

import os
import glob
import json
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO
from samgeo import SamGeo3


# ----------------------------------------------------------------------
# Ayarlar
# ----------------------------------------------------------------------
TEXT_PROMPT = "rooftop"      # "building" da denenebilir, karsilastir
IOU_THRESHOLD = 0.3          # YOLO kutusu <-> SAM3 text-mask eslesme esigi
YOLO_CONF = 0.25


def load_latest_yolo():
    runs_dir = r"C:\1.Projeler\yolo-sam\runs\detect"
    weight_files = glob.glob(os.path.join(runs_dir, "yolo11_geosam_model*", "weights", "best.pt"))
    if not weight_files:
        raise FileNotFoundError("Egitilmis YOLO agirligi (best.pt) bulunamadi!")
    latest_path = max(weight_files, key=os.path.getmtime)
    print(f"[+] YOLO Model Yuklendi: {latest_path}")
    return YOLO(latest_path)


def box_iou(box_a, box_b):
    """[x1,y1,x2,y2] formatinda iki kutu arasi IoU hesabi."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def mask_to_bbox(mask):
    """Binary mask (H,W) -> [x1,y1,x2,y2]. Bos maskede None doner."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def normalize_mask(mask):
    """SamGeo3 mask ciktisini (tensor/np, cesitli sekiller) 2D binary np array'e cevirir."""
    if hasattr(mask, "cpu"):
        mask = mask.squeeze().cpu().numpy()
    elif hasattr(mask, "numpy"):
        mask = mask.squeeze().numpy()
    else:
        mask = np.squeeze(np.asarray(mask))
    if mask.ndim > 2:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def match_text_masks_to_yolo_boxes(yolo_boxes, sam_masks, sam_boxes, iou_threshold):
    """
    Her YOLO kutusu icin en iyi ortusen SAM3 text-mask'i bulur.
    Eslesen maskeleri OR ile tek bir binary maskede birlestirir.

    Returns:
        combined_mask (np.ndarray | None), matched_count (int)
    """
    combined_mask = None
    matched_count = 0

    for y_box in yolo_boxes:
        best_iou = 0.0
        best_mask = None

        for s_mask, s_box in zip(sam_masks, sam_boxes):
            # s_box zaten SamGeo3 tarafindan mask'ten hesaplanmis olabilir;
            # tutarlilik icin maskeden yeniden hesapliyoruz.
            norm_mask = normalize_mask(s_mask)
            bbox = mask_to_bbox(norm_mask)
            if bbox is None:
                continue

            iou = box_iou(y_box, bbox)
            if iou > best_iou:
                best_iou = iou
                best_mask = norm_mask

        if best_mask is not None and best_iou >= iou_threshold:
            matched_count += 1
            if combined_mask is None:
                combined_mask = best_mask.copy()
            else:
                combined_mask = np.maximum(combined_mask, best_mask)

    return combined_mask, matched_count


if __name__ == "__main__":
    yolo_model = load_latest_yolo()

    # Text-mode icin enable_inst_interactivity=False (varsayilan) yeterli;
    # box+text ayni anda desteklenmedigi icin instance-interactivity acmiyoruz.
    sam = SamGeo3(backend="meta")

    val_img_dir = r"C:\1.Projeler\yolo-sam\building_data\val\images"
    output_dir = r"C:\1.Projeler\yolo-sam\samgeo3_text_ciktilari"
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    all_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    manifest_records = []

    for img_name in all_images:
        img_path = os.path.join(val_img_dir, img_name)

        # 1. YOLO ile kutu tespiti
        yolo_results = yolo_model.predict(source=img_path, conf=YOLO_CONF, save=False)[0]
        yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy().tolist()

        if not yolo_boxes:
            print(f"[{img_name}] YOLO hic kutu bulamadi, atlaniyor.")
            manifest_records.append({
                "image": img_name,
                "mask": None,
                "box_count": 0,
                "matched_count": 0,
                "mask_coverage_pct": 0.0,
                "status": "no_detection"
            })
            continue

        # 2. SAM3 text-mode ile TUM "rooftop" maskelerini uret
        sam.set_image(img_path)
        sam.generate_masks(TEXT_PROMPT, quiet=True)

        sam_masks = sam.masks if sam.masks is not None else []

        if len(sam_masks) == 0:
            print(f"[{img_name}] SAM3 text-mode hic maske bulamadi ('{TEXT_PROMPT}'), atlaniyor.")
            manifest_records.append({
                "image": img_name,
                "mask": None,
                "box_count": len(yolo_boxes),
                "matched_count": 0,
                "mask_coverage_pct": 0.0,
                "status": "no_text_mask"
            })
            continue

        # 3. YOLO kutulari <-> SAM3 text-mask'leri IoU ile eslestir
        combined_mask, matched_count = match_text_masks_to_yolo_boxes(
            yolo_boxes, sam_masks, sam.boxes, IOU_THRESHOLD
        )

        if combined_mask is None:
            print(f"[{img_name}] Hicbir SAM3 maskesi YOLO kutulariyla eslesmedi (IoU < {IOU_THRESHOLD}).")
            manifest_records.append({
                "image": img_name,
                "mask": None,
                "box_count": len(yolo_boxes),
                "matched_count": 0,
                "mask_coverage_pct": 0.0,
                "status": "no_match"
            })
            continue

        print(f"[{img_name}] YOLO {len(yolo_boxes)} kutu, {matched_count} tanesi SAM3 text-mask ile eslesti.")

        mask_filename = f"{os.path.splitext(img_name)[0]}_mask.png"
        out_mask_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(out_mask_path, (combined_mask * 255).astype(np.uint8))

        mask_coverage_pct = float(np.mean(combined_mask > 0) * 100)

        manifest_records.append({
            "image": img_name,
            "mask": mask_filename,
            "box_count": len(yolo_boxes),
            "matched_count": matched_count,
            "mask_coverage_pct": round(mask_coverage_pct, 2),
            "status": "ok"
        })

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "text_prompt": TEXT_PROMPT,
        "iou_threshold": IOU_THRESHOLD,
        "val_img_dir": val_img_dir,
        "total_images": len(all_images),
        "matched_ok_count": sum(1 for r in manifest_records if r["status"] == "ok"),
        "no_detection_count": sum(1 for r in manifest_records if r["status"] == "no_detection"),
        "no_text_mask_count": sum(1 for r in manifest_records if r["status"] == "no_text_mask"),
        "no_match_count": sum(1 for r in manifest_records if r["status"] == "no_match"),
        "records": manifest_records
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Manifest kaydedildi: {manifest_path}")
    print("[+] Tum gorseller islendi.")
