"""shadow_masks / uncertainty_masks / unseen_pixels_mask / tree_mask dosyalarının
gerçekte ne anlama geldiğini TAHMİN ETMEDEN, piksel değerlerine bakarak
doğrular. Eğitimden önce mutlaka çalıştırılmalı — loss maskeleme mantığı
buna bağlı.

extract_shadows.py bitmiş olmalı (bu script D:\\shadow-eo altındaki açılmış
dosyaları okuyor, hiçbir şey indirmiyor/açmıyor).
"""
import numpy as np
from PIL import Image
from pathlib import Path

BASE = Path(r"D:\shadow-eo")
TILE, IDX = "OMA_135", "6"  # daha önce gördüğümüz gerçek bir örnek

FILES = {
    "shadow_max (asıl gölge, DSM Max'ten)":
        BASE / f"shadow_max_compressed/shadow_masks/{TILE}/{TILE}_{IDX}_shadow.png",
    "shadow_min (asıl gölge, DSM Min'den)":
        BASE / f"shadows/shadow_masks/{TILE}/{TILE}_{IDX}_shadow.png",
    "uncertainty_mask (max)":
        BASE / f"shadow_max_compressed/uncertainty_masks/{TILE}/{TILE}_{IDX}_uncertainty_mask.png",
    "unseen_pixels_mask (max)":
        BASE / f"shadow_max_compressed/uncertainty_masks/{TILE}/{TILE}_{IDX}_unseen_pixels_mask.png",
    "tree_mask (max)":
        BASE / f"shadow_max_compressed/uncertainty_masks/{TILE}/{TILE}_{IDX}_tree_mask.png",
}

for label, path in FILES.items():
    if not path.exists():
        print(f"{label:45s} -> DOSYA YOK: {path}")
        continue
    arr = np.array(Image.open(path))
    unique, counts = np.unique(arr, return_counts=True)
    total = arr.size
    print(f"\n{label}")
    print(f"  boyut: {arr.shape}, dtype: {arr.dtype}")
    print(f"  benzersiz değerler (ilk 10): {unique[:10]}")
    for u, c in list(zip(unique, counts))[:10]:
        print(f"    değer={u}: {c} piksel ({100*c/total:.2f}%)")
