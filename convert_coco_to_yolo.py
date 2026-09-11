"""
Amac:
  train_v5/valid_v5/test_v5.json (COCO bbox formati) + images_merged
  klasorunu, Ultralytics YOLO'nun bekledigi yapiya donusturur:

    YOLO_ROOT/
      train/images/*.jpg|png   train/labels/*.txt
      val/images/*.jpg|png     val/labels/*.txt
      test/images/*.jpg|png    test/labels/*.txt
    data.yaml

  Her .txt icinde YOLO detection format satiri:
    class_id x_center_norm y_center_norm width_norm height_norm

  category_id=1 (COCO) -> class_id=0 (YOLO, tek sinif "building")

  Gorseller COPY edilir (images_merged bozulmaz), buyuk veri setinde
  disk alani ~2x kaplayacagini unutma.
"""

import json
import shutil
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------------

EXISTING_ROOT = Path(r"D:\building_segmentation_dataset_genisletilmis")
IMAGES_MERGED_DIR = EXISTING_ROOT / "images_merged"

INPUT_SUFFIX = "_v5"  # yolo egitimine bu versiyonu aktariyoruz

# YOLO formatli cikti, COCO setinden (images_merged + jsonlar) AYRI, kendi
# klasorunde D surucusunde tutulur - iki format birbirine karismasin diye
YOLO_ROOT = Path(r"D:\yolo_dataset_genisletilmis")

# internal split adi -> YOLO/Ultralytics'in bekledigi klasor adi
SPLIT_FOLDER_MAP = {
    "train": "train",
    "valid": "val",   # ultralytics convention: "val", data.yaml'inle uyumlu
    "test": "test",
}

CLASS_NAMES = {0: "building"}

# ---------------------------------------------------------------------------


def convert_split(split_name: str):
    json_path = EXISTING_ROOT / f"{split_name}{INPUT_SUFFIX}.json"
    if not json_path.exists():
        print(f"[{split_name}] HATA: {json_path} bulunamadi")
        return 0, 0

    coco = json.load(open(json_path, encoding="utf-8"))

    yolo_split = SPLIT_FOLDER_MAP[split_name]
    images_out = YOLO_ROOT / yolo_split / "images"
    labels_out = YOLO_ROOT / yolo_split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    n_images = 0
    n_labels = 0

    for img in coco["images"]:
        src_path = IMAGES_MERGED_DIR / img["file_name"]
        if not src_path.exists():
            print(f"  UYARI: kaynak gorsel yok, atlaniyor -> {img['file_name']}")
            continue

        width, height = img["width"], img["height"]
        if width <= 0 or height <= 0:
            with Image.open(src_path) as im:
                width, height = im.size

        dst_path = images_out / img["file_name"]
        shutil.copy2(src_path, dst_path)
        n_images += 1

        label_path = labels_out / (Path(img["file_name"]).stem + ".txt")
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            # 0-1 araligina clip et (kenar tasan bbox'lar icin guvenlik)
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # annotation'siz gorseller icin de BOS .txt yazilir (YOLO negatif ornek olarak kullanir)
        label_path.write_text("\n".join(lines), encoding="utf-8")
        n_labels += len(lines)

    print(f"[{split_name}] -> {yolo_split}: {n_images} gorsel, {n_labels} kutu yazildi")
    return n_images, n_labels


def write_data_yaml():
    yaml_path = YOLO_ROOT / "data.yaml"
    lines = [
        f"path: {YOLO_ROOT}",
        f"train: train/images",
        f"val: val/images",
        f"test: test/images",
        "names:",
    ]
    for cid, name in CLASS_NAMES.items():
        lines.append(f"  {cid}: {name}")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\ndata.yaml yazildi -> {yaml_path}")


def main():
    YOLO_ROOT.mkdir(parents=True, exist_ok=True)
    total_images = 0
    total_labels = 0
    for split_name in ("train", "valid", "test"):
        ni, nl = convert_split(split_name)
        total_images += ni
        total_labels += nl
    write_data_yaml()
    print(f"\nTOPLAM: {total_images} gorsel, {total_labels} kutu")
    print(f"\nSimdi egitim.py icindeki data= satirini su sekilde guncelle:")
    print(f'  data=r"{YOLO_ROOT / "data.yaml"}"')


if __name__ == "__main__":
    main()
