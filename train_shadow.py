"""S-EO -> Shadow U-Net eğitimi (makale: arXiv:2504.06920, Bölüm 3.2).

Mimari: U-Net + ResNet34 encoder (ImageNet ön-eğitimli) + SCSE (squeeze-and-
excitation) decoder attention — makalenin "squeeze and excitation blocks"
tanımına karşılık gelen segmentation_models_pytorch parametresi.

Optimizer: AdamWScheduleFree, lr=5e-3 (makaledeki değer).
Loss: Focal loss, ignore mask uygulanmış (min-max anlaşmazlığı + ağaç +
DSM boşluğu bölgeleri loss'a hiç girmiyor).

GPU NOTU: Makale batch_size=32 kullanıyor ama RTX 4060 8GB'da 512x512 +
ResNet34 + batch 32 VRAM'e sığmayabilir. Bunun yerine küçük gerçek batch
(REAL_BATCH_SIZE) + gradient accumulation ile efektif 32'ye ulaşıyoruz —
matematiksel olarak eşdeğer, sadece daha yavaş.

Kullanım:
    python train.py
Devam ettirmek için:
    python train.py --resume runs/shadow_unet/last.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader

from shadow_eo_dataset import ShadowEOStreamDataset, build_tile_split

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

RUN_DIR = Path(r"D:\shadow-eo\runs\shadow_unet")
PATCH_SIZE = 512
EFFECTIVE_BATCH_SIZE = 32          # makaledeki değer
REAL_BATCH_SIZE = 8                # RTX 4060 8GB için gerçekçi üst sınır
ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // REAL_BATCH_SIZE  # 4 adımda bir güncelle
# Encoder (ImageNet ön-eğitimli, zaten iyi öğrenilmiş özellikler) ve
# decoder+segmentation_head (sıfırdan öğreniyor) AYRI öğrenme oranı kullanır.
# Tek/yüksek bir LR ile ikisini birlikte güncellemek, encoder'ın ön-eğitimli
# bilgisini bozarak epoch 8-19 ve 9-12 arasında gözlenen recall çöküşünü
# (model giderek "her yeri arkaplan de" diyen dejenere çözüme kayması)
# tetikleyen ana sebep olarak değerlendirildi.
ENCODER_LR = 1e-4                  # düşük — ön-eğitimli ağırlıkları korumak için
DECODER_LR = 1e-3                  # yüksek — sıfırdan öğrenen kısım
STEPS_PER_EPOCH = 500              # IterableDataset'te "epoch" kavramı yok,
                                    # bu yüzden sabit adım sayısıyla tanımlıyoruz
VAL_STEPS = 60
N_EPOCHS = 40
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75  
                    
# Dice, focal loss'a EK olarak toplanıyor (loss = focal + DICE_WEIGHT * dice).
# Dice "şeklin ne kadar örtüştüğüne" bakar (BCE/focal gibi piksel-piksel
# ortalama almaz), bu yüzden modelin "hiç gölge tahmin etme" gibi dejenere
# bir çözüme kaymasını -- recall çöküşlerinde gördüğümüz patern -- daha sert
# cezalandırır: tahmin boşsa Dice loss neredeyse maksimuma çıkar.
DICE_WEIGHT = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0   # streaming tar okuması worker'lar arası kolay paylaşılmıyor;
                   # 0 en güvenlisi, darboğaza girerse ayrı bir yaklaşım gerekir


# ---------------------------------------------------------------------------
# Focal loss — ignore mask destekli
# ---------------------------------------------------------------------------

def focal_loss_with_ignore(logits: torch.Tensor, target: torch.Tensor,
                            ignore: torch.Tensor, gamma: float = FOCAL_GAMMA,
                            alpha: float = FOCAL_ALPHA) -> torch.Tensor:
    """logits: (B,1,H,W) ham model çıktısı. target: (B,H,W) 0/1.
    ignore: (B,H,W) bool, True olan piksel loss'a girmez.

    ignore edilen piksel oranı görüntüden görüntüye değişiyor (bkz. veri
    doğrulama testleri, %20-85 arası); bu yüzden normalize ederken TÜM batch
    genelinde geçerli piksel sayısına bölüyoruz, tek görüntünün piksel
    sayısına değil — yoksa çok ignore edilen bir görüntü loss'u yapay olarak
    şişirir/söndürür.
    """
    logits = logits.squeeze(1)  # (B,H,W)
    valid = ~ignore
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = alpha_t * (1 - p_t).pow(gamma) * bce

    loss = loss[valid]
    if loss.numel() == 0:
        # Bu batch'te hiç geçerli piksel yoksa (nadiren, hepsi ignore),
        # sıfır loss dön — geri yayılım hiçbir şeyi bozmaz.
        return logits.sum() * 0.0
    return loss.mean()


def dice_loss_with_ignore(logits: torch.Tensor, target: torch.Tensor,
                           ignore: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss, ignore mask hariç tutularak, TÜM batch üzerinde
    (tek görüntü değil) toplanmış piksellerle hesaplanıyor -- focal loss'un
    normalize ediliş mantığıyla tutarlı, ve tek bir görüntüde az sayıda
    geçerli piksel kalırsa Dice'ın yapay şekilde uçlara savrulmasını önlüyor.
    """
    logits = logits.squeeze(1)
    valid = ~ignore
    p = torch.sigmoid(logits)

    p_valid = p[valid]
    target_valid = target[valid]
    if p_valid.numel() == 0:
        return logits.sum() * 0.0

    intersection = (p_valid * target_valid).sum()
    denom = p_valid.sum() + target_valid.sum()
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice


def combined_loss(logits: torch.Tensor, target: torch.Tensor,
                   ignore: torch.Tensor) -> torch.Tensor:
    """loss = focal + DICE_WEIGHT * dice. Eğitim ve validasyonda AYNI
    fonksiyon çağrılıyor, ikisi arasında tutarsızlık olmasın diye."""
    focal = focal_loss_with_ignore(logits, target, ignore)
    dice = dice_loss_with_ignore(logits, target, ignore)
    return focal + DICE_WEIGHT * dice


# ---------------------------------------------------------------------------
# Metrikler — precision / recall / IoU / F1, ignore mask hariç tutularak
# ---------------------------------------------------------------------------

METRIC_THRESHOLD = 0.3


@torch.no_grad()
def confusion_counts(logits: torch.Tensor, target: torch.Tensor,
                      ignore: torch.Tensor, threshold: float = METRIC_THRESHOLD
                      ) -> tuple[int, int, int, int]:
    """Bir batch için (tp, fp, fn, tn) — ignore edilen pikseller sayılmaz.
    Sonuç Python int (tensor değil), epoch boyunca kolayca toplanabilsin diye.
    """
    logits = logits.squeeze(1)
    pred = (torch.sigmoid(logits) >= threshold)
    target_bool = target.bool()
    valid = ~ignore

    tp = (pred & target_bool & valid).sum().item()
    fp = (pred & ~target_bool & valid).sum().item()
    fn = (~pred & target_bool & valid).sum().item()
    tn = (~pred & ~target_bool & valid).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-7) -> dict:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "iou": iou, "f1": f1}


# ---------------------------------------------------------------------------
# Tek adımlık (belirli sayıda batch) DataLoader sarmalayıcı
# ---------------------------------------------------------------------------

def make_loader(tiles: set[str], batch_size: int, shuffle_buffer: int) -> DataLoader:
    ds = ShadowEOStreamDataset(tiles, patch_size=PATCH_SIZE,
                                shuffle_buffer=shuffle_buffer)
    return DataLoader(ds, batch_size=batch_size, num_workers=NUM_WORKERS,
                       pin_memory=(DEVICE == "cuda"))


def cycle(loader: DataLoader):
    """IterableDataset her tükendiğinde (tüm arşiv bir kere okununca) baştan
    başlatır — sonsuz bir akış gibi davranır, epoch sınırını STEPS_PER_EPOCH
    belirler."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Eğitim / validasyon döngüleri
# ---------------------------------------------------------------------------

def run_epoch(model, optimizer, train_iter, n_steps: int, accum_steps: int) -> float:
    model.train()
    optimizer.train()  # AdamWScheduleFree'nin kendine özgü modu
    optimizer.zero_grad()
    total_loss = 0.0
    for step in range(n_steps):
        image, target, ignore = next(train_iter)
        image, target, ignore = image.to(DEVICE), target.to(DEVICE), ignore.to(DEVICE)

        logits = model(image)
        loss = combined_loss(logits, target, ignore) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            print(f"    adım {step+1}/{n_steps}  ort. loss: {total_loss/(step+1):.4f}")
    return total_loss / n_steps


@torch.no_grad()
def run_validation(model, val_iter, n_steps: int) -> dict:
    model.eval()
    total_loss = 0.0
    tp_sum = fp_sum = fn_sum = tn_sum = 0
    for _ in range(n_steps):
        image, target, ignore = next(val_iter)
        image, target, ignore = image.to(DEVICE), target.to(DEVICE), ignore.to(DEVICE)
        logits = model(image)
        loss = combined_loss(logits, target, ignore)
        total_loss += loss.item()

        tp, fp, fn, tn = confusion_counts(logits, target, ignore)
        tp_sum += tp; fp_sum += fp; fn_sum += fn; tn_sum += tn

    result = metrics_from_counts(tp_sum, fp_sum, fn_sum, tn_sum)
    result["loss"] = total_loss / n_steps
    return result


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cihaz: {DEVICE}")

    train_tiles, val_tiles = build_tile_split()
    print(f"Train tile: {len(train_tiles)}, Val tile: {len(val_tiles)}")

    train_loader = make_loader(train_tiles, REAL_BATCH_SIZE, shuffle_buffer=200)
    val_loader = make_loader(val_tiles, REAL_BATCH_SIZE, shuffle_buffer=50)
    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        decoder_attention_type="scse",  # makaledeki squeeze-and-excitation
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    optimizer = AdamWScheduleFree([
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},
        {"params": list(model.decoder.parameters())
                    + list(model.segmentation_head.parameters()), "lr": DECODER_LR},
    ])

    start_epoch = 0
    best_val_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Checkpoint'te farklı sayıda param group varsa (örn. eski,
            # tek-gruplu bir optimizer'dan geliyorsa) yukarıdaki satır
            # ValueError fırlatır — bu normal, aşağıdaki except onu yakalar.
            for group, lr in zip(optimizer.param_groups, (ENCODER_LR, DECODER_LR)):
                group["lr"] = lr
            print("  optimizer durumu (momentum vb.) checkpoint'ten yüklendi")
        except ValueError:
            print("  UYARI: checkpoint'teki optimizer, farklı param group "
                  "yapısından (muhtemelen encoder/decoder LR ayrımından önceki "
                  "eski koddan) geldiği için YÜKLENEMEDİ. Sadece model "
                  "ağırlıkları alındı, optimizer (momentum) sıfırdan başlıyor "
                  "— bu ağırlıkları bozmaz, sadece ilk birkaç adımda "
                  "optimizasyon biraz daha yavaş ısınabilir.")
        start_epoch = ckpt["epoch"] + 1
        # ESKİ checkpoint'ler best_val_loss tutuyordu, best_val_iou değil.
        # -1.0'dan başlatıyoruz ki resume sonrası ilk epoch, IoU'su ne
        # olursa olsun "yeni en iyi" olarak kaydedilsin — kaldığın yerden
        # itibaren en iyi IoU'yu yeniden izlemeye başlıyoruz.
        print(f"'{args.resume}'den devam ediliyor, epoch {start_epoch}'dan başlanıyor")

    print(f"Efektif batch: {EFFECTIVE_BATCH_SIZE} (gerçek={REAL_BATCH_SIZE} x "
          f"accum={ACCUM_STEPS}), encoder_lr={ENCODER_LR}, decoder_lr={DECODER_LR}")

    for epoch in range(start_epoch, N_EPOCHS):
        t0 = time.time()
        print(f"\n=== Epoch {epoch+1}/{N_EPOCHS} ===")

        train_loss = run_epoch(model, optimizer, train_iter, STEPS_PER_EPOCH, ACCUM_STEPS)

        optimizer.eval()  # AdamWScheduleFree: değerlendirmeden önce şart
        val_metrics = run_validation(model, val_iter, VAL_STEPS)
        val_loss = val_metrics["loss"]

        dt = time.time() - t0
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"IoU={val_metrics['iou']:.4f}  precision={val_metrics['precision']:.4f}  "
              f"recall={val_metrics['recall']:.4f}  F1={val_metrics['f1']:.4f}  "
              f"süre={dt/60:.1f} dk")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_iou": best_val_iou,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, RUN_DIR / "last.pt")
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            ckpt["best_val_iou"] = best_val_iou
            torch.save(ckpt, RUN_DIR / "best.pt")
            print(f"  -> yeni en iyi model kaydedildi (IoU={best_val_iou:.4f}, "
                  f"val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()