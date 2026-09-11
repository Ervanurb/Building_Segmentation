"""Kullanicinin kendi yukledigi TIF dosyalari icin onizleme, sinir hesabi,
AOI kirpma (canli haritayla ayni GSD'ye normalize ederek) ve bozuk/eksik
coğrafi referansi onarma fonksiyonlari."""
import numpy as np
import rasterio
import rasterio.warp as rasterio_warp
import rasterio.windows as rio_windows
from rasterio.enums import Resampling as RioResampling


def build_tif_preview_array(tif_path, max_dim=1500):
    """Yuklenen TIF icin haritada gosterilecek kucuk RGB onizleme dizisi uretir."""
    with rasterio.open(tif_path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))
        band_count = min(src.count, 3)
        arr = src.read(
            indexes=list(range(1, band_count + 1)),
            out_shape=(band_count, out_h, out_w),
            resampling=RioResampling.bilinear,
        )
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return arr


def get_tif_bounds_4326(tif_path):
    """TIF dosyasinin EPSG:4326 (lat/lon) sinirlarini doner: (minx, miny, maxx, maxy).

    Cozunurluk artirma (upscaling/super-resolution) araclarindan gecirilmis
    goruntuler cogunlukla CRS/affine transform bilgisini kaybeder ya da yeni
    piksel boyutuna gore olceklemez - bu durumda harita onizlemesi bozuk bir
    alana zoom yapar ve Draw araciyla secim yapilamaz gibi gorunur. Bu yuzden
    burada acik ve anlasilir bir hata firlatiyoruz, sessizce gecmiyoruz.
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise RuntimeError(
                "Bu TIF dosyasında coğrafi referans (CRS) bilgisi yok. Görsel muhtemelen "
                "bir çözünürlük artırma / süper-çözünürlük aracından geçirilirken coğrafi "
                "etiketler kayboldu. Orijinal (işlem görmemiş) TIF'in CRS ve transform "
                "bilgisini yeniden yazmanız gerekiyor."
            )
        minx, miny, maxx, maxy = rasterio_warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        if not all(np.isfinite(v) for v in (minx, miny, maxx, maxy)) or minx >= maxx or miny >= maxy:
            raise RuntimeError(
                "TIF dosyasının coğrafi sınırları (bounds) geçersiz/anlamsız görünüyor. "
                "Görsel muhtemelen bir çözünürlük artırma / süper-çözünürlük aracından "
                "geçirilirken transform bilgisi bozulmuş (piksel sayısı değişmiş ama "
                "coğrafi ölçek güncellenmemiş)."
            )
        return (minx, miny, maxx, maxy)


def crop_uploaded_tif_to_aoi(src_path, bbox_4326, out_path, dst_crs="EPSG:3857", target_gsd_m=None):
    """
    Yuklenen (kullanicinin kendi) TIF dosyasindan, EPSG:4326 bbox'una karsilik gelen
    alani kirpar, gerekirse 8-bit RGB'ye cevirir ve metrik bir CRS'ye (varsayilan
    EPSG:3857 - canli haritadaki indirme akisiyla ayni) yeniden projekte ederek
    ayri bir GeoTIFF olarak kaydeder. Boylece run_detection_pipeline() hicbir
    degisiklik yapilmadan bu dosyayi da isleyebilir.

    target_gsd_m verilirse, cikti bu piksel boyutuna (m/piksel) yeniden orneklenir.
    Bu onemli: YOLO/SAM3 modelleri canli haritadaki sabit zoom (~0.30 m/px, zoom19)
    olceginde calisiyor - kullanicinin TIF'i cok farkli bir cozunurlukteyse (ornegin
    cok yuksek cozunurluklu drone goruntusu ya da cok dusuk cozunurluklu uydu
    goruntusu), YOLO kutulari bulunsa bile SAM3'un urettigi maskeler modelin
    alistigi olcekten sapip minimum boyut esiginin (SAM3_MIN_MASK_PX) altinda
    kalabilir ve tamamen elenebilir. Ayni olcege sabitlemek bu sorunu giderir.
    """
    minx, miny, maxx, maxy = bbox_4326

    with rasterio.open(src_path) as src:
        src_minx, src_miny, src_maxx, src_maxy = rasterio_warp.transform_bounds(
            "EPSG:4326", src.crs, minx, miny, maxx, maxy
        )
        window = rio_windows.from_bounds(
            src_minx, src_miny, src_maxx, src_maxy, transform=src.transform
        )
        full_window = rio_windows.Window(0, 0, src.width, src.height)
        window = window.intersection(full_window)
        if window.width < 2 or window.height < 2:
            raise RuntimeError("Secilen alan yuklenen TIF dosyasinin sinirlari disinda kaliyor.")

        # Kesirli piksel pencerelerinde src.read() ile gercekte donen dizi boyutu,
        # window.width/height'in dogrudan int()'e cevrilmesinden farkli olabiliyor
        # (rasterio okurken yuvarlama yapiyor). Bu kaymayi onlemek icin pencereyi
        # once tam piksele yuvarlayip (tum istenen alani kapsayacak sekilde disari
        # dogru), sonra o TAM boyutla okuyoruz - boylece data.shape ile win_w/win_h
        # her zaman birebir eslesiyor.
        window = window.round_offsets(op="floor").round_lengths()
        window = window.intersection(full_window)
        win_w = int(window.width)
        win_h = int(window.height)

        win_transform = src.window_transform(window)
        band_count = min(src.count, 3)
        data = src.read(
            indexes=list(range(1, band_count + 1)),
            window=window,
            out_shape=(band_count, win_h, win_w),
            boundless=True,
            fill_value=0,
        )

        if data.dtype != np.uint8:
            data = data.astype(np.float32)
            lo, hi = np.percentile(data, 2), np.percentile(data, 98)
            if hi <= lo:
                hi = lo + 1.0
            data = np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

        if band_count == 1:
            data = np.repeat(data, 3, axis=0)

        src_bounds = rasterio.transform.array_bounds(win_h, win_w, win_transform)

        dst_transform, dst_width, dst_height = rasterio_warp.calculate_default_transform(
            src.crs, dst_crs, win_w, win_h, *src_bounds,
            resolution=(target_gsd_m, target_gsd_m) if target_gsd_m else None,
        )
        dst_data = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
        for b in range(3):
            rasterio_warp.reproject(
                source=data[b],
                destination=dst_data[b],
                src_transform=win_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=RioResampling.bilinear,
            )

        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=dst_height, width=dst_width, count=3,
            dtype=np.uint8, crs=dst_crs, transform=dst_transform,
        ) as dst:
            dst.write(dst_data)

    return out_path

def repair_missing_georeference(reference_tif_path, broken_tif_path, out_path):
    """
    broken_tif_path dosyasinin CRS/transform bilgisi eksik ya da gecersizse (ornegin
    bir cozunurluk artirma / super-cozunurluk isleminden sonra), reference_tif_path
    (ayni goruntunun islem gormemis, coğrafi referansi saglam hali) dosyasindan CRS
    ve piksel olcegini alip broken_tif_path'in yeni piksel boyutuna gore olcekleyerek
    out_path'e yazar. Piksel degerleri degismez, sadece coğrafi etiketler eklenir.
    """
    with rasterio.open(reference_tif_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_w, ref_h = ref.width, ref.height

    if ref_crs is None:
        raise RuntimeError("Referans TIF'te de CRS bilgisi bulunamadı.")

    with rasterio.open(broken_tif_path) as broken:
        broken_w, broken_h = broken.width, broken.height
        profile = broken.profile.copy()
        data = broken.read()

    scale_x = ref_w / broken_w
    scale_y = ref_h / broken_h
    new_transform = ref_transform * ref_transform.scale(scale_x, scale_y)

    profile.update(crs=ref_crs, transform=new_transform)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    return out_path

