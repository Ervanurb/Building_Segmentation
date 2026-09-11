"""
YOLO formatındaki (.txt) etiketleri COCO JSON formatına çevirir.

Desteklenen YOLO formatları (satır başına otomatik algılanır):
  - Detection (bbox):       class x_center y_center width height       (4 değer)
  - Segmentation (polygon): class x1 y1 x2 y2 x3 y3 ...                (6+ değer, çift sayı)
  Tüm koordinatlar normalize (0-1 arası) olmalı - standart YOLO formatı.

Çıktı, standart COCO JSON şemasında (images / annotations / categories),
prepare_solarfinder.py'nin okuduğu pycocotools ile uyumlu formatta üretilir.

Beklenen giriş klasör yapısı:
    images_dir/
        img_0001.jpg
        ...
    labels_dir/
        img_0001.txt   # aynı isim, YOLO formatında etiketler
        ...

Kullanım:
  python yolo_to_coco.py \
      --images_dir "C:\\1.Projeler\\yolo-sam\\building_data\\train\\images" \
      --labels_dir "C:\\1.Projeler\\yolo-sam\\building_data\\train\\labels" \
      --output_json "C:\\1.Projeler\\yolo-sam\\building_data\\train\\_annotations.coco.json" \
      --data_yaml "C:\\1.Projeler\\yolo-sam\\data.yaml"

  # data_yaml verilmezse, karşılaşılan class_id'ler "class_0", "class_1" gibi
  # otomatik isimlendirilir.
"""
import argparse
import json
import os
import glob
import cv2

try:
    import yaml
except ImportError:
    yaml = None


def parse_yolo_label_line(line: str):
    """
    Tek bir YOLO etiket satırını ayrıştırır.
    Dönen: (class_id, kind, values)
      kind = "bbox"    -> values = (x_center, y_center, w, h) normalize
      kind = "polygon" -> values = [x1,y1, x2,y2, ...] normalize (düz liste)
    Geçersiz/bozuk satırlarda None döner.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        return class_id, "bbox", coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        return class_id, "polygon", coords
    else:
        return None


def load_class_names(data_yaml_path):
    """data.yaml'dan sınıf isimlerini okur. Bulunamazsa None döner."""
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return None
    if yaml is None:
        print("  ⚠️ PyYAML kurulu değil, data.yaml okunamıyor (pip install pyyaml). Otomatik isimlendirme kullanılacak.")
        return None

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        return None
    # names, dict {0: 'building', 1: 'pool'} ya da list ['building', 'pool'] olabilir
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return None


def yolo_to_coco(images_dir: str, labels_dir: str, output_json: str, class_names=None):
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"'{images_dir}' altında görüntü bulunamadı.")

    images_out = []
    annotations_out = []
    encountered_class_ids = set()

    image_id = 1
    annotation_id = 1
    skipped_no_label = 0
    bbox_count = 0
    polygon_count = 0

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️ Görüntü okunamadı, atlanıyor: {img_path}")
            continue
        h, w = img.shape[:2]

        images_out.append({
            "id": image_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        if not os.path.exists(label_path):
            skipped_no_label += 1
            image_id += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parsed = parse_yolo_label_line(line)
            if parsed is None:
                continue
            class_id, kind, coords = parsed
            encountered_class_ids.add(class_id)

            if kind == "bbox":
                x_c, y_c, bw, bh = coords
                # COCO bbox formatı: [x_min, y_min, width, height] (piksel, mutlak)
                px_w = bw * w
                px_h = bh * h
                px_x = (x_c * w) - px_w / 2
                px_y = (y_c * h) - px_h / 2
                bbox_coco = [round(px_x, 2), round(px_y, 2), round(px_w, 2), round(px_h, 2)]
                area = round(px_w * px_h, 2)
                segmentation = []  # bbox-only etiketlerde segmentation boş bırakılır
                bbox_count += 1
            else:  # polygon
                px_points = []
                xs, ys = [], []
                for i in range(0, len(coords), 2):
                    px = coords[i] * w
                    py = coords[i + 1] * h
                    px_points.extend([round(px, 2), round(py, 2)])
                    xs.append(px)
                    ys.append(py)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox_coco = [round(x_min, 2), round(y_min, 2),
                             round(x_max - x_min, 2), round(y_max - y_min, 2)]
                # Shoelace formülüyle poligon alanı
                area = 0.0
                n = len(xs)
                for i in range(n):
                    j = (i + 1) % n
                    area += xs[i] * ys[j] - xs[j] * ys[i]
                area = round(abs(area) / 2.0, 2)
                segmentation = [px_points]
                polygon_count += 1

            annotations_out.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox_coco,
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0,
            })
            annotation_id += 1

        image_id += 1

    # Kategori listesi: data.yaml varsa oradan, yoksa karşılaşılan ID'lerden otomatik üret
    if class_names:
        categories_out = [
            {"id": cid, "name": name, "supercategory": "none"}
            for cid, name in sorted(class_names.items())
        ]
    else:
        categories_out = [
            {"id": cid, "name": f"class_{cid}", "supercategory": "none"}
            for cid in sorted(encountered_class_ids)
        ]

    coco_dict = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    return {
        "num_images": len(images_out),
        "num_annotations": len(annotations_out),
        "skipped_no_label": skipped_no_label,
        "bbox_count": bbox_count,
        "polygon_count": polygon_count,
        "categories": categories_out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="YOLO görüntülerinin bulunduğu klasör")
    p.add_argument("--labels_dir", required=True, help="YOLO .txt etiketlerinin bulunduğu klasör")
    p.add_argument("--output_json", required=True, help="Üretilecek COCO JSON dosyasının yolu")
    p.add_argument("--data_yaml", default=None,
                    help="(Opsiyonel) YOLO data.yaml yolu - sınıf isimlerini buradan okur. "
                         "Verilmezse sınıflar 'class_0', 'class_1' şeklinde otomatik isimlendirilir.")
    args = p.parse_args()

    class_names = load_class_names(args.data_yaml)
    if class_names:
        print(f"Sınıf isimleri data.yaml'dan okundu: {class_names}")
    else:
        print("Sınıf isimleri otomatik üretilecek (class_0, class_1, ...).")

    stats = yolo_to_coco(args.images_dir, args.labels_dir, args.output_json, class_names)

    print(f"\nToplam görüntü: {stats['num_images']}")
    print(f"Toplam annotation: {stats['num_annotations']}")
    print(f"Etiket dosyası bulunamayan görüntü: {stats['skipped_no_label']}")
    print(f"Bbox tipi annotation: {stats['bbox_count']}")
    print(f"Polygon tipi annotation: {stats['polygon_count']}")
    print(f"Kategoriler: {stats['categories']}")
    print(f"\nÇıktı: {args.output_json}")

    if stats["bbox_count"] > 0 and stats["polygon_count"] == 0:
        print(
            "\nNot: Etiketler bbox formatında olduğu için JSON'daki 'segmentation' "
            "alanları boş bırakıldı (sadece 'bbox' dolu). Piksel-hassas poligon "
            "gerekiyorsa kaynak etiketlerin segmentasyon formatında olması gerekir."
        )


if __name__ == "__main__":
    main()
