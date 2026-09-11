import os
import glob
import json
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO
from samgeo import SamGeo3


def load_latest_yolo():
    runs_dir = r"C:\1.Projeler\yolo-sam\runs\detect"
    weight_files = glob.glob(os.path.join(runs_dir, "yolo11_geosam_model*", "weights", "best.pt"))
    if not weight_files:
        raise FileNotFoundError("Eğitilmiş YOLO ağırlığı (best.pt) bulunamadı!")
    latest_path = max(weight_files, key=os.path.getmtime)
    print(f"[+] YOLO Model Yüklendi: {latest_path}")
    return YOLO(latest_path)


if __name__ == '__main__':
    yolo_model = load_latest_yolo()
    sam = SamGeo3(backend="meta", enable_inst_interactivity=True)

    val_img_dir = r"C:\1.Projeler\yolo-sam\building_data\val\images"
    output_dir = r"C:\1.Projeler\yolo-sam\samgeo3_ciktilari"
    mask_dir = os.path.join(output_dir, "masks")  # YENİ: kalıcı ham maske klasörü
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    all_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    manifest_records = []  # YENİ: Streamlit'in okuyacağı kayıt listesi

    for i, img_name in enumerate(all_images):
        img_path = os.path.join(val_img_dir, img_name)

        # 1. YOLO ile kutu tespiti
        yolo_results = yolo_model.predict(source=img_path, conf=0.25, save=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy().tolist()

        if not boxes:
            print(f"[{img_name}] YOLO hiç kutu bulamadı, atlanıyor.")
            manifest_records.append({
                "image": img_name,
                "mask": None,
                "box_count": 0,
                "mask_coverage_pct": 0.0,
                "status": "no_detection"
            })
            continue

        print(f"[{img_name}] YOLO {len(boxes)} kutu tespit etti. SAM3 maske üretiyor...")

        # 2. SAM3 ile maske üret -> ARTIK KALICI KLASÖRE KAYDEDİLİYOR
        sam.set_image(img_path)
        mask_filename = f"{os.path.splitext(img_name)[0]}_mask.png"
        out_mask_path = os.path.join(mask_dir, mask_filename)
        sam.generate_masks_by_boxes_inst(
            boxes=boxes,
            box_crs=None,
            output=out_mask_path,
            dtype="uint8",
            multimask_output=False,
        )

        # --- GÖRSEL OLUŞTURMA (OVERLAY) BÖLÜMÜ KALDIRILDI ---
        # Overlay artık Streamlit tarafında, ham maskeden anlık üretilecek.

        # Manifest için maske kapsama yüzdesini hesapla
        mask = cv2.imread(out_mask_path, cv2.IMREAD_UNCHANGED)
        if mask is not None and mask.ndim == 3:
            mask = mask[..., 0]
        mask_coverage_pct = float(np.mean(mask > 0) * 100) if mask is not None and np.any(mask > 0) else 0.0

        manifest_records.append({
            "image": img_name,
            "mask": mask_filename,
            "box_count": len(boxes),
            "mask_coverage_pct": round(mask_coverage_pct, 2),
            "status": "ok"
        })

        # NOT: maske artık SİLİNMİYOR, kalıcı olarak masks/ klasöründe duruyor

    # YENİ: manifest.json'ı yaz
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "val_img_dir": val_img_dir,
        "total_images": len(all_images),
        "detected_count": sum(1 for r in manifest_records if r["status"] == "ok"),
        "no_detection_count": sum(1 for r in manifest_records if r["status"] == "no_detection"),
        "records": manifest_records
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Manifest kaydedildi: {manifest_path}")
    print("\n[+] Tüm görseller işlendi ve maskeler masks/ klasörüne kaydedildi!")



    