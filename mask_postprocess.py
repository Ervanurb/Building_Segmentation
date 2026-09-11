"""
Ortak maske son-işleme (post-processing) modülü.

Bu modül, SAM3/samgeo'dan gelen ham rooftop maskelerini temizlemek için
kullanılan tüm fonksiyonları TEK bir yerde toplar. Streamlit'e (app_live.py)
hiçbir bağımlılığı yoktur -- saf cv2/numpy/geopandas/rasterio/samgeo işlemleri
içerir, bu yüzden hem canlı Streamlit akışında hem offline scriptlerde
(compare_shadow_rooftop.py, prepare_geosam3_masks.py, vb.) aynen kullanılabilir.

Kullanım:
    from mask_postprocess import clean_and_simplify_mask, raster_mask_to_gdf

    # Sadece piksel-seviyeli maske temizliği (morfolojik kapama + kontur sadeleştirme)
    cleaned = clean_and_simplify_mask(mask_uint8_255, epsilon_factor=0.01)

    # Ham binary maskeden coğrafi GeoDataFrame'e (isteğe bağlı regularize ile)
    gdf = raster_mask_to_gdf(mask_bin, transform, crs, apply_regularize=True)

Neden ayrı bir dosya:
    Önceden bu mantık hem app_live.py hem compare_shadow_rooftop.py içinde ayrı
    ayrı (ve tutarsız şekilde) tekrarlanıyordu. Bu modül tek kaynak haline
    getirir -- bir düzeltme yapıldığında (örn. regularize() parametreleri
    değişince) tek yerden güncellenir, iki script de otomatik senkron kalır.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np
import geopandas as gpd
from rasterio.features import shapes as rio_shapes


# ---------------------------------------------------------------------------
# 1. Piksel-seviyeli temizlik: morfolojik kapama + kontur sadeleştirme
# ---------------------------------------------------------------------------
def clean_and_simplify_mask(
    mask: np.ndarray,
    epsilon_factor: float = 0.015,
    min_area: int = 50,
) -> np.ndarray:
    """SAM maskesindeki piksel-seviyeli tırtıkları/gürültüyü temizler ve düzleştirir.

    Bu, rasterio.features.shapes() ile poligona çevrilmeden ÖNCE, ham binary
    maske üzerinde çalışan hafif bir temizlik adımıdır. Bina şeklini korur,
    sadece merdiven-basamağı (staircase) kenarları ve küçük gürültü
    parçacıklarını giderir. Ana yönlere/dik açılara OTURTMAZ -- bunun için
    regularize_gdf() / raster_mask_to_gdf(apply_regularize=True) kullan.

    Args:
        mask: 2D uint8 ikili maske (değerleri 0 ve 255).
        epsilon_factor: Kontur sadeleştirme toleransı (cv2.approxPolyDP).
            0.003 civarı: sadece piksel-merdiveni giderir, şekli korur.
            0.015 civarı: daha görünür düzleştirme (varsayılan sidebar değeri).
            0.01 - 0.03 arası tipik kullanım aralığıdır.
        min_area: Bu alanın (piksel cinsinden) altındaki konturlar atlanır
            (tek piksellik gürültü parçacıklarını eler).

    Returns:
        Temizlenmiş ve düzleştirilmiş binary maske (aynı boyutta, 0/255 uint8).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    simplified_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon = epsilon_factor * perimeter

        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

        cv2.drawContours(simplified_mask, [approx], -1, 255, thickness=cv2.FILLED)

    return simplified_mask


# ---------------------------------------------------------------------------
# 2. Bina-geometrisi farkında düzeltme: samgeo.common.regularize()
# ---------------------------------------------------------------------------
def regularize_gdf(gdf: gpd.GeoDataFrame, **regularize_kwargs) -> gpd.GeoDataFrame:
    """Bina poligonlarını samgeo.common.regularize() ile dik açılara/ana yönlere oturtur.

    clean_and_simplify_mask()'ten kalitatif olarak farklıdır: kontur
    sadeleştirmesi değil, GERÇEK bina geometrisi varsayımıyla (dikdörtgen /
    ana yön) çalışır. Piksel-merdiveni gürültüsünü çok daha güçlü şekilde
    giderir.

    NOT: regularize()'ın tam parametre imzası samgeo sürümüne göre değişebilir
    (bu yazıda doğrulanamadı -- GitHub robots.txt ham dosya erişimini
    engelledi). Bu yüzden hata durumunda exception'ı YUTMUYORUZ burada;
    çağıran taraf (örn. app_live.py) kendi bağlamında (Streamlit uyarısı,
    log, vb.) ele almalı. Kendi ortamında `help(regularize)` çalıştırıp
    gerçek imzayı doğrulaman önerilir.

    Args:
        gdf: Bina poligonlarını içeren GeoDataFrame (EPSG:4326 veya projekte CRS).
        **regularize_kwargs: samgeo.common.regularize()'a olduğu gibi iletilir
            (örn. parallel_threshold, allow_diagonals, allow_circles, vb. --
            kendi samgeo sürümünde `help(regularize)` ile doğrula).

    Returns:
        Regularize edilmiş GeoDataFrame.

    Raises:
        ImportError: samgeo kurulu değilse.
        Exception: regularize() çağrısı sırasında oluşan herhangi bir hata
            (parametre uyuşmazlığı dahil) olduğu gibi yükseltilir.
    """
    from samgeo.common import regularize  # local import: samgeo ağır bir bağımlılık

    if len(gdf) == 0:
        return gdf

    return regularize(gdf, **regularize_kwargs)


# ---------------------------------------------------------------------------
# 3. Uçtan uca: ham binary raster maske -> temiz coğrafi GeoDataFrame
# ---------------------------------------------------------------------------
def raster_mask_to_gdf(
    mask_bin: np.ndarray,
    transform,
    crs,
    apply_simplify: bool = False,
    simplify_epsilon: float = 0.015,
    apply_regularize: bool = False,
    regularize_kwargs: dict | None = None,
    orig_box_4326=None,
    baseline_epsilon: float | None = 0.003,
) -> gpd.GeoDataFrame:
    """Ham binary raster maskeyi (SAM3/samgeo çıktısı) temiz bir coğrafi
    GeoDataFrame'e çevirir. app_live.py ve compare_shadow_rooftop.py'nin
    ortak kullanacağı TEK giriş noktasıdır.

    İşlem sırası:
        1. (varsa) baseline_epsilon ile HER ZAMAN hafif piksel-merdiveni temizliği
        2. (apply_simplify=True ise) simplify_epsilon ile ek/daha güçlü düzleştirme
        3. rasterio.features.shapes() ile poligona çevirme (kaynak CRS'te, örn. EPSG:3857)
        4. (apply_regularize=True ise) samgeo.common.regularize() ile dik açıya oturtma
           -- BU ADIM projekte/metre CRS'teyken yapılır, EPSG:4326'ya çevrilmeden
           ÖNCE, çünkü regularize()'ın eşik parametreleri (parallel_threshold vb.)
           metre birimini varsayar; derece biriminde çalıştırılırsa poligonları
           boşaltabilir.
        5. EPSG:4326'ya projeksiyon
        6. (orig_box_4326 verilmişse) sadece kesişen poligonları filtreleme

    Args:
        mask_bin: 2D binary maske (0/1 veya 0/255, otomatik algılanır).
        transform: rasterio Affine transform (görüntü -> coğrafi koordinat).
        crs: Maskenin CRS'i (örn. "EPSG:3857").
        apply_simplify: clean_and_simplify_mask() ile EK düzleştirme uygulansın mı.
        simplify_epsilon: apply_simplify=True iken kullanılacak epsilon.
        apply_regularize: samgeo.common.regularize() uygulansın mı.
        regularize_kwargs: regularize()'a iletilecek ek parametreler.
        orig_box_4326: Verilirse, sadece bu kutuyla kesişen poligonlar tutulur
            (EPSG:4326, shapely geometry). Poligonların kendisi KIRPILMAZ.
        baseline_epsilon: HER ZAMAN uygulanan hafif temel temizlik (None ise kapalı).

    Returns:
        EPSG:4326'ya projekte edilmiş GeoDataFrame.
    """
    working_mask = (mask_bin > 0).astype(np.uint8)

    if baseline_epsilon is not None:
        cleaned_255 = clean_and_simplify_mask(
            working_mask * 255, epsilon_factor=baseline_epsilon, min_area=20
        )
        working_mask = (cleaned_255 > 0).astype(np.uint8)

    if apply_simplify:
        simplified_255 = clean_and_simplify_mask(
            working_mask * 255, epsilon_factor=simplify_epsilon
        )
        working_mask = (simplified_255 > 0).astype(np.uint8)

    polygons = [
        {"geometry": geom}
        for geom, _val in rio_shapes(working_mask, mask=working_mask.astype(bool), transform=transform)
    ]

    # DİKKAT: regularize() burada, henüz projekte (metre cinsinden, örn. EPSG:3857)
    # CRS'teyken çağrılıyor -- EPSG:4326'ya (derece) çevirmeden ÖNCE. Sebep:
    # regularize()'ın parallel_threshold / diagonal_threshold_reduction gibi
    # eşik parametreleri metre birimini varsayıyor olabilir; derece biriminde
    # (EPSG:4326) çalıştırılırsa eşikler anlamsız kalıp poligonları
    # boşaltabilir (bu, "regularize seçilince tüm maskeler gitti" sorununun
    # kök nedeni olabilir).
    gdf = gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": p["geometry"], "properties": {}} for p in polygons],
        crs=crs,
    )

    if apply_regularize and len(gdf) > 0:
        pre_count = len(gdf)
        try:
            regularized = regularize_gdf(gdf, **(regularize_kwargs or {}))

            if regularized is None:
                warnings.warn(
                    "regularize() None döndürdü, ham/simplify uygulanmış poligonlar "
                    "kullanılıyor.",
                    stacklevel=2,
                )
            elif len(regularized) == 0:
                warnings.warn(
                    f"regularize() {pre_count} poligonu 0'a düşürdü (muhtemelen "
                    f"içsel bir filtre/eşik nedeniyle), ham/simplify uygulanmış "
                    f"poligonlar kullanılıyor.",
                    stacklevel=2,
                )
            else:
                if regularized.crs is None:
                    regularized = regularized.set_crs(crs, allow_override=True)

                if len(regularized) < pre_count:
                    warnings.warn(
                        f"regularize() {pre_count} poligondan {len(regularized)} "
                        f"tanesini korudu ({pre_count - len(regularized)} bina "
                        f"muhtemelen içsel bir filtre/eşik nedeniyle elendi).",
                        stacklevel=2,
                    )
                gdf = regularized
        except Exception as e:
            warnings.warn(
                f"regularize() çalıştırılamadı, ham/simplify uygulanmış poligonlar "
                f"kullanılıyor. Sebep: {e}. Kendi ortamında `help(regularize)` ile "
                f"gerçek parametre imzasını doğrula.",
                stacklevel=2,
            )

    gdf = gdf.to_crs("EPSG:4326")

    if orig_box_4326 is not None and len(gdf) > 0:
        gdf = gdf[gdf.geometry.intersects(orig_box_4326)].reset_index(drop=True)

    return gdf