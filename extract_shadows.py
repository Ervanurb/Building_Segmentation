"""shadows_max.tar.gz ve shadows_min.tar.gz'yi D:\\shadow-eo altına açar.

pansharpened_crops_color_corrected'a DOKUNMUYOR — o hâlâ akış (streaming)
yoluyla, hiç açılmadan okunacak. Sadece küçük shadow arşivlerini (toplam
~4.5 GB sıkıştırılmış) gerçekten diske açıyoruz, çünkü onlar disk sorunu
yaratmayacak kadar küçük ve eğitim sırasında rastgele erişim (random access)
gerektiriyorlar (streaming'de rastgele patch/tile seçmek zor olurdu).

Windows'un yerleşik tar.exe'si yerine Python'un tarfile modülünü kullanıyoruz
— daha önce tar.exe'nin (asıl sebebi disk doluluğuydu ama) hataya düştüğünü
gördük, bu daha şeffaf ve güvenilir.
"""
import tarfile
from pathlib import Path

BASE = Path(r"D:\shadow-eo")
DEST = BASE  # arşivlerin içindeki yol önekleri (shadow_max_compressed/, shadows/)
             # zaten birbirinden ayrışıyor, aynı köke açmak sorun değil


def extract_with_progress(archive_path: Path, dest: Path, progress_every=2000):
    print(f"\n=== Açılıyor: {archive_path.name} ===")
    n = 0
    with tarfile.open(archive_path, mode="r:gz") as tf:
        for member in tf:
            tf.extract(member, path=dest, filter="data")
            n += 1
            if n % progress_every == 0:
                print(f"  ... {n} dosya açıldı")
    print(f"  Bitti: {n} dosya -> {dest}")


if __name__ == "__main__":
    extract_with_progress(BASE / "shadows_max.tar.gz", DEST)
    extract_with_progress(BASE / "shadows_min.tar.gz", DEST)
    print("\nTüm shadow arşivleri açıldı.")
