"""Eğitilmiş Shadow U-Net'i S-EO validasyon (Omaha ağırlıklı) görsellerinde
çalıştırıp tahmini GERÇEK gölge maskesiyle (shadow_max/min agreement) yan
yana gösterir, her görsel için gerçek IoU'yu hesaplar.

shadow_eo_dataset.py ile aynı klasörde olmalı (ondan tar-akış ve split
mantığını import ediyoruz, kod tekrarı olmasın diye).

Kullanım:
    python analyze_predictions_seo.py
"""
import io
import tarfile
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image

from shadow_eo_dataset import (
    BASE, PANSHARPENED_PARTS, NAME_RE, ChainedParts, build_tile_split, _load_mask,
)

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path(r"D:\shadow-eo\runs\shadow_unet\last_epoch_iyi_recall.pt")
OUTPUT_PATH = Path(r"D:\shadow-eo\prediction_check_seo.png")
N_IMAGES = 5
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path):
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        decoder_attention_type="scse", in_channels=3, classes=1,
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch'] + 1}, val_loss={ckpt['val_loss']:.4f}")
    if ckpt.get("val_metrics"):
        m = ckpt["val_metrics"]
        print(f"  (eğitim sırasındaki, tüm val seti üzerinden) "
              f"IoU={m['iou']:.4f}  recall={m['recall']:.4f}")
    return model


# ---------------------------------------------------------------------------
# Bir görüntü için: RGB (tar'dan) + gerçek hedef/ignore (diskten), TAM boyut
# (build_sample'daki gibi rastgele crop YOK — tüm görüntüyü görmek istiyoruz)
# ---------------------------------------------------------------------------

def ground_truth_for(tile: str, idx: str):
    shadow_max = _load_mask(BASE / f"shadow_max_compressed/shadow_masks/{tile}/{tile}_{idx}_shadow.png")
    shadow_min = _load_mask(BASE / f"shadows/shadow_masks/{tile}/{tile}_{idx}_shadow.png")
    unseen = _load_mask(BASE / f"shadow_max_compressed/uncertainty_masks/{tile}/{tile}_{idx}_unseen_pixels_mask.png")
    tree = _load_mask(BASE / f"shadow_max_compressed/uncertainty_masks/{tile}/{tile}_{idx}_tree_mask.png")
    if shadow_max is None or shadow_min is None or unseen is None or tree is None:
        return None, None

    target_max = 1 - (shadow_max > 0).astype(np.uint8)
    target_min = 1 - (shadow_min > 0).astype(np.uint8)
    agree = target_max == target_min
    ignore = (~agree) | (unseen > 0) | (tree > 0)
    return target_max, ignore


def crop_to_multiple_of_32(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[: (h // 32) * 32, : (w // 32) * 32]


# ---------------------------------------------------------------------------
# Val tile'larından ilk N_IMAGES uygun örneği tar'dan akıtarak topla
# ---------------------------------------------------------------------------

def collect_val_samples(val_tiles: set[str], n: int):
    """İlk n uygun görüntüyü değil, n FARKLI tile'dan birer görüntü toplar.

    Aksi halde tar içinde bir tile'ın ~20-40 görüntüsü art arda durduğu için
    (aynı 500x500m bölgenin komşu/örtüşen kırpımları), ilk n görüntü genelde
    tek bir tile'dan gelir ve hepsi görsel olarak neredeyse aynı mahalleyi
    gösterir — çeşitlilik göremeyiz.
    """
    samples = []
    seen_tiles: set[str] = set()
    chained = ChainedParts(PANSHARPENED_PARTS)
    with tarfile.open(fileobj=chained, mode="r|gz") as tf:
        for member in tf:
            if len(samples) >= n:
                break
            if not member.isfile():
                continue
            m = NAME_RE.search(member.name)
            if not m:
                continue
            tile, idx = m.group(1), m.group(2)
            if tile not in val_tiles or tile in seen_tiles:
                continue
            target, ignore = ground_truth_for(tile, idx)
            if target is None:
                continue
            f = tf.extractfile(member)
            try:
                rgb_arr = np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))
            except Exception:  # noqa: BLE001
                continue
            if rgb_arr.shape[:2] != target.shape[:2]:
                continue
            samples.append((f"{tile}_{idx}", rgb_arr, target, ignore))
            seen_tiles.add(tile)
            print(f"  bulundu: {tile}_{idx}  ({len(samples)}/{n})")
    return samples


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main():
    model = load_model(CHECKPOINT_PATH)
    _train_tiles, val_tiles = build_tile_split()

    print(f"\n{N_IMAGES} validasyon görseli tar'dan aranıyor "
          f"(arşivin başından itibaren taranıyor, biraz sürebilir)...")
    samples = collect_val_samples(val_tiles, N_IMAGES)
    if not samples:
        print("UYARI: hiç uygun örnek bulunamadı.")
        return

    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]

    for row, (name, rgb_arr, target, ignore) in enumerate(samples):
        rgb_arr = crop_to_multiple_of_32(rgb_arr)
        target = crop_to_multiple_of_32(target)
        ignore = crop_to_multiple_of_32(ignore)

        tensor = torch.from_numpy(rgb_arr.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        pred = (prob >= THRESHOLD).astype(np.uint8)

        valid = ~ignore
        tp = int(((pred == 1) & (target == 1) & valid).sum())
        fp = int(((pred == 1) & (target == 0) & valid).sum())
        fn = int(((pred == 0) & (target == 1) & valid).sum())
        iou = tp / (tp + fp + fn + 1e-7)

        axes[row, 0].imshow(rgb_arr)
        axes[row, 0].set_title(f"{name}\nRGB")

        gt_display = target.astype(float)
        gt_display[ignore] = 0.5  # ignore edilen bölgeler gri görünsün
        axes[row, 1].imshow(gt_display, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Gerçek (gri=ignore edilen)")

        axes[row, 2].imshow(prob, cmap="inferno", vmin=0, vmax=1)
        axes[row, 2].set_title("Tahmin olasılığı")

        overlay = rgb_arr.astype(float) / 255.0
        overlay[pred == 1] = overlay[pred == 1] * 0.4 + np.array([1, 0, 0]) * 0.6
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title(f"Tahmin (kırmızı) — bu görsel IoU={iou:.3f}")

        for ax in axes[row]:
            ax.axis("off")

        print(f"  {name}: IoU={iou:.4f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120)
    print(f"\nKaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()