"""S-EO -> yolo-sam Shadow U-Net eğitim verisi.

DOĞRULANMIŞ BİLGİLER (tahminle değil, gerçek veriden ölçülerek):
- shadow_max.png / shadow_min.png: 0 = GÖLGE, 1 = aydınlık (ters kodlu!)
- unseen_pixels_mask.png: DSM'den projeksiyon almayan pikseller (makaledeki
  "uncertainty mask" / Algorithm 2'deki U). Loss'tan hariç tutulur.
- tree_mask.png: NDVI-tabanlı vegetation mask. Loss'tan hariç tutulur.
- uncertainty_mask.png: makalede tanımlanmamış üçüncü bir dosya, KULLANILMIYOR.

EĞİTİM STRATEJİSİ (makale, Bölüm 3.2 "Min-Max shadow masks"):
- Hedef: shadow_max VE shadow_min'in AYNI FİKİRDE olduğu piksellerde gölge
  etiketi (ikisi aynı olduğu için hangisi kullanılırsa kullanılsın fark etmez).
- Ignore mask = (max != min) VEYA unseen_pixels_mask VEYA tree_mask.
  Bu bölgelerdeki pikseller loss'a hiç girmez.

DİSK KULLANIMI:
- pansharpened_crops_color_corrected: HİÇ DİSKE AÇILMADI, tar'dan akış
  (streaming) olarak okunuyor — 86 GB'lık disk sorununu bu şekilde aştık.
- shadow_max / shadow_min / uncertainty_masks: küçük oldukları için diske
  açıldı (extract_shadows.py), rastgele erişim (random access) buradan.

TRAIN/VAL SPLIT (makale, Bölüm "Experiments"):
- Train: neredeyse tüm Jacksonville (JAX_341/342/335/334 hariç) + San Diego'dan
  50 site.
- Val/Test: geri kalan her şey (Omaha'nın tamamı dahil).
- Makale "hangi 50 San Diego tile'ı" demiyor -> burada SEED'li deterministik
  bir seçim yapılıyor (SPLIT_SEED). İstersen değiştirilebilir.
"""
from __future__ import annotations

import io
import random
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import IterableDataset

BASE = Path(r"D:\shadow-eo")
SPLIT_SEED = 42
N_UCSD_TRAIN_TILES = 50
EXCLUDED_JAX_TILES = {"JAX_341", "JAX_342", "JAX_335", "JAX_334"}

PANSHARPENED_PARTS = [
    BASE / f"pansharpened_crops_color_corrected.part.tar.gz.{suf}"
    for suf in ("aa", "ab", "ac", "ad")
]

# pansharpened_he/OMA_135/OMA_135_6_rgb.png -> ("OMA_135", "6")
NAME_RE = re.compile(r"pansharpened_he/([A-Z]+_\d+)/\1_(\d+)_rgb\.png$")


# ---------------------------------------------------------------------------
# Train/val split — curated_aois_v3.csv'den şehir bazında
# ---------------------------------------------------------------------------

def build_tile_split() -> tuple[set[str], set[str]]:
    """(train_tiles, val_tiles) — tile ID'leri (örn. "JAX_12", "OMA_135")."""
    df = pd.read_csv(BASE / "curated_aois_v3.csv")
    # CSV'nin gerçek kolon adını görmeden varsayım yapmamak için esnek arıyoruz.
    id_col = next(c for c in df.columns if "id" in c.lower() or "tile" in c.lower() or "aoi" in c.lower())
    tile_ids = df[id_col].astype(str).tolist()

    jax = sorted(t for t in tile_ids if t.startswith("JAX_") and t not in EXCLUDED_JAX_TILES)
    oma = sorted(t for t in tile_ids if t.startswith("OMA_"))
    ucsd = sorted(t for t in tile_ids if t.startswith("UCSD_"))

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ucsd)
    ucsd_train = set(ucsd[:N_UCSD_TRAIN_TILES])
    ucsd_val = set(ucsd[N_UCSD_TRAIN_TILES:])

    train_tiles = set(jax) | ucsd_train
    val_tiles = set(oma) | ucsd_val
    return train_tiles, val_tiles


# ---------------------------------------------------------------------------
# pansharpened arşivini diske hiç yazmadan akış olarak okuma
# ---------------------------------------------------------------------------

class ChainedParts:
    """4 parçayı diske birleştirmeden, sırayla okuyan sahte tek dosya."""

    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.fh = open(self.paths[0], "rb")

    def read(self, size=-1):
        data = self.fh.read(size)
        while (size < 0 or len(data) < size) and self.idx + 1 < len(self.paths):
            self.idx += 1
            self.fh.close()
            self.fh = open(self.paths[self.idx], "rb")
            more = self.fh.read(-1 if size < 0 else size - len(data))
            data += more
            if size < 0:
                break
        return data


# ---------------------------------------------------------------------------
# Örnek üretimi: RGB (akıştan) + shadow_max/min + unseen/tree (diskten)
# ---------------------------------------------------------------------------

def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.array(Image.open(path))
    except Exception:  # noqa: BLE001 -- bozuk bir PNG tüm eğitimi durdurmasın,
                        # bu görüntü sadece atlanır (72k+ dosyalık bir arşivde
                        # birkaç bozuk dosya beklenir)
        return None


def build_sample(tile: str, idx: str, rgb_arr: np.ndarray, patch_size: int):
    """Bir görüntü için (image, target, ignore) çifti üretir, veya eksik
    dosya/uygun olmayan boyut varsa None döner (çağıran atlar)."""
    shadow_max = _load_mask(BASE / f"shadow_max_compressed/shadow_masks/{tile}/{tile}_{idx}_shadow.png")
    shadow_min = _load_mask(BASE / f"shadows/shadow_masks/{tile}/{tile}_{idx}_shadow.png")
    unseen_max = _load_mask(BASE / f"shadow_max_compressed/uncertainty_masks/{tile}/{tile}_{idx}_unseen_pixels_mask.png")
    tree_max = _load_mask(BASE / f"shadow_max_compressed/uncertainty_masks/{tile}/{tile}_{idx}_tree_mask.png")
    if shadow_max is None or shadow_min is None or unseen_max is None or tree_max is None:
        return None

    h, w = shadow_max.shape[:2]
    if rgb_arr.shape[0] != h or rgb_arr.shape[1] != w:
        return None  # boyut uyuşmazlığı -> güvenli atla
    if h < patch_size or w < patch_size:
        return None

    # 0=gölge,1=aydınlık ham kodlamayı standart 1=gölge,0=aydınlık'a çeviriyoruz.
    target_max = 1 - (shadow_max > 0).astype(np.uint8)
    target_min = 1 - (shadow_min > 0).astype(np.uint8)

    agree = target_max == target_min
    ignore = (~agree) | (unseen_max > 0) | (tree_max > 0)
    target = target_max  # agreement bölgesinde max==min zaten, fark etmez

    # Rastgele 512x512 patch (RGB + target + ignore aynı pencereden kesilir)
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    sl = (slice(top, top + patch_size), slice(left, left + patch_size))

    image = rgb_arr[sl]
    target = target[sl]
    ignore = ignore[sl]

    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    target_t = torch.from_numpy(target).float()
    ignore_t = torch.from_numpy(ignore.astype(np.uint8)).bool()
    return image_t, target_t, ignore_t


# ---------------------------------------------------------------------------
# IterableDataset — pansharpened'ı bir kere baştan sona akıtır, shuffle buffer
# ---------------------------------------------------------------------------

class ShadowEOStreamDataset(IterableDataset):
    def __init__(self, allowed_tiles: set[str], patch_size: int = 512,
                 shuffle_buffer: int = 200, samples_per_image: int = 1):
        super().__init__()
        self.allowed_tiles = allowed_tiles
        self.patch_size = patch_size
        self.shuffle_buffer = shuffle_buffer
        self.samples_per_image = samples_per_image

    def __iter__(self):
        buffer = []
        chained = ChainedParts(PANSHARPENED_PARTS)
        with tarfile.open(fileobj=chained, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                m = NAME_RE.search(member.name)
                if not m:
                    continue
                tile, idx = m.group(1), m.group(2)
                if tile not in self.allowed_tiles:
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue
                try:
                    rgb_arr = np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))
                except Exception:  # noqa: BLE001 -- bozuk bir PNG tüm epoch'u durdurmasın
                    continue

                for _ in range(self.samples_per_image):
                    sample = build_sample(tile, idx, rgb_arr, self.patch_size)
                    if sample is None:
                        continue
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        i = random.randrange(len(buffer))
                        buffer[i], buffer[-1] = buffer[-1], buffer[i]
                        yield buffer.pop()
        random.shuffle(buffer)
        yield from buffer


if __name__ == "__main__":
    train_tiles, val_tiles = build_tile_split()
    print(f"Train tile sayısı: {len(train_tiles)}  (örnek: {sorted(train_tiles)[:5]})")
    print(f"Val tile sayısı:   {len(val_tiles)}  (örnek: {sorted(val_tiles)[:5]})")

    ds = ShadowEOStreamDataset(train_tiles, patch_size=512, shuffle_buffer=50)
    it = iter(ds)

    N_SAMPLES = 15
    shadow_ratios, ignore_ratios = [], []
    for i in range(N_SAMPLES):
        image, target, ignore = next(it)
        shadow_ratios.append(target.mean().item() * 100)
        ignore_ratios.append(ignore.float().mean().item() * 100)
        print(f"  crop {i+1:2d}: gölge={shadow_ratios[-1]:5.2f}%  "
              f"ignore={ignore_ratios[-1]:5.2f}%")

    print(f"\n{N_SAMPLES} crop ortalaması:")
    print(f"  gölge oranı  (ort): {sum(shadow_ratios)/N_SAMPLES:.2f}%")
    print(f"  ignore oranı (ort): {sum(ignore_ratios)/N_SAMPLES:.2f}%")
    print(f"  tensor boyutu: image={tuple(image.shape)}, target={tuple(target.shape)}, "
          f"ignore={tuple(ignore.shape)}")