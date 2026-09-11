"""
SAM3 (rooftop) maskeleri ile Shadow U-Net maskelerini kıyaslar.

samgeo3_ciktilari/manifest.json içindeki her görüntü için:
  1. Aynı görüntü üzerinde shadow U-Net çalıştırılır (taze tahmin, kayıtlı
     bir shadow maskesi dosyası aranmaz).
  2. SAM3'ün ürettiği rooftop maskesi (masks/{img}_mask.png) okunur.
  3. rooftop_mask AND NOT shadow_mask işlemiyle gölgesiz temiz maske üretilir.
  4. Sonuç cleaned_masks/ klasörüne kaydedilir, istatistikler
     manifest_with_shadow.json içine yazılır.

Kullanım:

python compare_shadow_rooftop.py --samgeo_output_dir "C:\1.Projeler\yolo-sam\samgeo3_ciktilari" --shadow_checkpoint "C:\1.Projeler\yolo-sam\shadow_unet\shadow_unet_v2_tekrar.pth" --shadow_unet_dir "C:\1.Projeler\yolo-sam\shadow_unet"
  python compare_shadow_rooftop.py \
      --samgeo_output_dir "C:\\1.Projeler\\yolo-sam\\samgeo3_ciktilari" \
      --shadow_checkpoint "C:\\1.Projeler\\yolo-sam\\shadow_unet\\shadow_unet_finetuned.pth" \
      --shadow_unet_dir "C:\\1.Projeler\\yolo-sam\\shadow_unet"


       komut:   python compare_shadow_rooftop.py --samgeo_output_dir "C:\1.Projeler\yolo-sam\samgeo3_ciktilari" --shadow_checkpoint "C:\1.Projeler\yolo-sam\shadow_unet\shadow_unet_finetuned.pth" --shadow_unet_dir "C:\1.Projeler\yolo-sam\shadow_unet"
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_shadow_model(shadow_unet_dir: str, checkpoint_path: str, device: torch.device,
                        encoder: str = "resnet18"):
    """shadow_unet/model.py içindeki build_shadow_unet'i import edip checkpoint yükler."""
    if shadow_unet_dir not in sys.path:
        sys.path.insert(0, shadow_unet_dir)
    from model import build_shadow_unet  # shadow_unet/model.py

    model = build_shadow_unet(encoder_name=encoder, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_shadow_mask(model, image_path: str, device: torch.device,
                          image_size: int = 512, threshold: float = 0.5) -> np.ndarray:
    """Verilen görüntü üzerinde shadow U-Net çalıştırır, orijinal boyuta resize edilmiş
    binary maske (0/255, uint8) döndürür."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)       # activation=None -> ham logit
        probs = torch.sigmoid(logits)       # 0-1 arası olasılığa çevir

    pred_mask = probs.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h))
    pred_binary = (pred_mask > threshold).astype(np.uint8) * 255
    return pred_binary


def remove_shadow_overlap(rooftop_mask: np.ndarray, shadow_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """rooftop_mask AND NOT shadow_mask. shadow_overlap.py ile aynı mantık."""
    if rooftop_mask.shape != shadow_mask.shape:
        raise ValueError(
            f"Maske boyutları uyuşmuyor: rooftop {rooftop_mask.shape} "
            f"vs shadow {shadow_mask.shape}."
        )

    rooftop_bin = (rooftop_mask > 0)
    shadow_bin = (shadow_mask > 0)

    rooftop_pixel_count = int(rooftop_bin.sum())
    cleaned_bin = rooftop_bin & (~shadow_bin)
    removed_pixel_count = rooftop_pixel_count - int(cleaned_bin.sum())
    removed_ratio = (
        removed_pixel_count / rooftop_pixel_count
        if rooftop_pixel_count > 0 else 0.0
    )

    cleaned_mask = (cleaned_bin.astype(np.uint8) * 255)

    stats = {
        "rooftop_pixel_count": rooftop_pixel_count,
        "removed_pixel_count": removed_pixel_count,
        "removed_ratio": round(removed_ratio, 4),
    }
    return cleaned_mask, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samgeo_output_dir", required=True,
                    help="prepare_geosam3_masks.py'nin çıktı klasörü (manifest.json burada)")
    p.add_argument("--shadow_checkpoint", required=True,
                    help="Shadow U-Net .pth dosyası")
    p.add_argument("--shadow_unet_dir", required=True,
                    help="model.py'nin bulunduğu shadow_unet klasörü")
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    manifest_path = os.path.join(args.samgeo_output_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    val_img_dir = manifest["val_img_dir"]
    mask_dir = os.path.join(args.samgeo_output_dir, "masks")
    cleaned_dir = os.path.join(args.samgeo_output_dir, "cleaned_masks")
    os.makedirs(cleaned_dir, exist_ok=True)

    print("Shadow U-Net yükleniyor...")
    model = load_shadow_model(args.shadow_unet_dir, args.shadow_checkpoint, device, args.encoder)

    updated_records = []
    for rec in manifest["records"]:
        if rec["status"] != "ok" or rec["mask"] is None:
            updated_records.append(rec)
            continue

        img_path = os.path.join(val_img_dir, rec["image"])
        rooftop_mask_path = os.path.join(mask_dir, rec["mask"])

        rooftop_mask = cv2.imread(rooftop_mask_path, cv2.IMREAD_UNCHANGED)
        if rooftop_mask is not None and rooftop_mask.ndim == 3:
            rooftop_mask = rooftop_mask[..., 0]
        if rooftop_mask is None:
            print(f"[{rec['image']}] rooftop maskesi okunamadı, atlanıyor.")
            updated_records.append(rec)
            continue

        shadow_mask = predict_shadow_mask(
            model, img_path, device,
            image_size=args.image_size, threshold=args.threshold,
        )

        cleaned_mask, stats = remove_shadow_overlap(rooftop_mask, shadow_mask)

        cleaned_filename = f"{os.path.splitext(rec['image'])[0]}_cleaned.png"
        cleaned_path = os.path.join(cleaned_dir, cleaned_filename)
        cv2.imwrite(cleaned_path, cleaned_mask)

        rec["cleaned_mask"] = cleaned_filename
        rec["shadow_removed_pixel_count"] = stats["removed_pixel_count"]
        rec["shadow_removed_ratio"] = stats["removed_ratio"]
        updated_records.append(rec)

        print(f"[{rec['image']}] gölge nedeniyle çıkarılan: %{stats['removed_ratio']*100:.2f}")

    manifest["records"] = updated_records
    out_manifest_path = os.path.join(args.samgeo_output_dir, "manifest_with_shadow.json")
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nTamamlandı. Güncellenmiş manifest: {out_manifest_path}")
    print(f"Temiz maskeler: {cleaned_dir}")


if __name__ == "__main__":
    main()