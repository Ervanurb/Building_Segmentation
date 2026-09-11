"""Rooftop/building footprint regularization for the yolo-sam pipeline.
Added: Minimum thickness / sliver polygon filtering for seam and shadow cuts.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate as shapely_rotate

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_SIMPLIFY_TOLERANCE = 1.5
DEFAULT_MIN_EDGE_LENGTH = 2.0
DEFAULT_DIAGONAL_WINDOW_DEG = 0.0
DEFAULT_MIN_KEEP_IOU = 0.50

DEFAULT_SIMPLIFY_TOLERANCE_M = 1.2
DEFAULT_MIN_EDGE_LENGTH_M = 3.0

# Yeni: Metre cinsinden kabul edilebilir minimum bina et kalınlığı/en dar kenarı.
# Bir binanın en dar cephesi (saçak/çatı kıymığı değilse) nadiren 2.5m altındadır.
DEFAULT_MIN_THICKNESS_M = 2.5


def is_sliver_or_too_narrow(geom: Polygon, min_thickness_m: float = DEFAULT_MIN_THICKNESS_M) -> bool:
    """Geometrinin minimum sınırlayıcı dikdörtgenini (oriented bounding box)
    çıkarıp en kısa kenarını kontrol eder. 
    Kıymıklar (sliver), sınırda kesilmiş ince şeritler ve gölge artıkları True döner.
    """
    if geom is None or geom.is_empty or geom.area <= 0:
        return True
    try:
        mrr = geom.minimum_rotated_rectangle
        if mrr.geom_type != "Polygon":
            return True
        coords = list(mrr.exterior.coords)
        if len(coords) < 4:
            return True
        
        edge_lengths = [
            math.hypot(coords[i][0] - coords[i + 1][0], coords[i][1] - coords[i + 1][1])
            for i in range(len(coords) - 1)
        ]
        min_edge = min(edge_lengths)
        return min_edge < min_thickness_m
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Step 1a: mask -> simplified polygon, PIXEL space
# ---------------------------------------------------------------------------

def mask_to_polygon(
    mask: np.ndarray,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
) -> Optional[Polygon]:
    import cv2

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return None
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None

    poly = Polygon(largest.reshape(-1, 2))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None

    simplified = poly.simplify(simplify_tolerance, preserve_topology=True)
    if simplified.is_empty:
        return poly
    if simplified.geom_type != "Polygon":
        return poly
    return simplified


# ---------------------------------------------------------------------------
# Step 1b: mask -> georeferenced polygon (Filtreler Entegre Edildi)
# ---------------------------------------------------------------------------

def instance_mask_to_polygon(
    inst_mask: np.ndarray,
    transform,
    min_area_m2: float = 1.0,
    min_thickness_m: float = DEFAULT_MIN_THICKNESS_M,
) -> Optional[Polygon | MultiPolygon]:
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    mask_bin = (np.asarray(inst_mask) > 0)
    if mask_bin.sum() == 0:
        return None

    polys = []
    for geom, _val in rio_shapes(mask_bin.astype(np.uint8), mask=mask_bin,
                                  transform=transform):
        poly = shapely_shape(geom)
        if not poly.is_valid or poly.is_empty or poly.area < min_area_m2:
            continue
        
        # Dış sınır bazlı poligon
        candidate = Polygon(poly.exterior)
        if not candidate.is_valid:
            candidate = candidate.buffer(0)

        # V yarıklarını ve çatlakları kapat (Closing)
        candidate = candidate.buffer(2.5).buffer(-2.5)
        if candidate.is_empty or not candidate.is_valid:
            continue
        # Eğer buffer sonrası geometri çok parçalıya bölünürse en büyük parçayı koru
        if candidate.geom_type == "MultiPolygon":
            candidate = max(candidate.geoms, key=lambda a: a.area)


        # Kıymık/ince şerit kontrolü
        if is_sliver_or_too_narrow(candidate, min_thickness_m=min_thickness_m):
            continue

        polys.append(candidate)

    if not polys:
        return None
    return unary_union(polys) if len(polys) > 1 else polys[0]


def regularize_instance_mask_ex(
    inst_mask: np.ndarray,
    transform,
    *,
    min_area_m2: float = 1.0,
    min_thickness_m: float = DEFAULT_MIN_THICKNESS_M,
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOLERANCE_M,
    min_keep_iou: float = DEFAULT_MIN_KEEP_IOU,
    min_edge_length_m: float = DEFAULT_MIN_EDGE_LENGTH_M,
    diagonal_window_deg: float = DEFAULT_DIAGONAL_WINDOW_DEG,
) -> tuple[Optional[Polygon | MultiPolygon], bool]:
    raw = instance_mask_to_polygon(
        inst_mask, transform, min_area_m2=min_area_m2, min_thickness_m=min_thickness_m
    )
    if raw is None:
        return None, False
        
    if raw.geom_type == "MultiPolygon":
        # MultiPolygon içinde kalan parçaları tek tek filtrele
        valid_geoms = [
            g for g in raw.geoms 
            if g.area >= min_area_m2 and not is_sliver_or_too_narrow(g, min_thickness_m)
        ]
        if not valid_geoms:
            return None, False
        return (valid_geoms[0] if len(valid_geoms) == 1 else MultiPolygon(valid_geoms)), False

    if raw.geom_type != "Polygon":
        return raw, False
    # Aşırı sivri/dışarı taşan tekil köşeleri budamak için:
    # Çok küçük bir pozitif-negatif buffer yerine poligonu sadeleştirme toleransını artırın:
    simplified = raw.simplify(1.0, preserve_topology=True) # 0.6 yerine 1.0 metre tolerans

    simplified = raw.simplify(simplify_tolerance_m, preserve_topology=True)
    if simplified.is_empty or simplified.geom_type != "Polygon":
        simplified = raw
        
    reg_poly, changed = regularize_polygon(
        simplified, min_keep_iou=min_keep_iou, min_edge_length=min_edge_length_m,
        diagonal_window_deg=diagonal_window_deg)
        
    # Regularize edildikten sonra şekil bozulup ince şerite dönüştü mü kontrolü
    if is_sliver_or_too_narrow(reg_poly, min_thickness_m=min_thickness_m):
        return None, False

    return reg_poly, changed


def regularize_instance_mask(
    inst_mask: np.ndarray,
    transform,
    *,
    min_area_m2: float = 1.0,
    min_thickness_m: float = DEFAULT_MIN_THICKNESS_M,
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOLERANCE_M,
    min_keep_iou: float = DEFAULT_MIN_KEEP_IOU,
    min_edge_length_m: float = DEFAULT_MIN_EDGE_LENGTH_M,
    diagonal_window_deg: float = DEFAULT_DIAGONAL_WINDOW_DEG,
) -> Optional[Polygon]:
    poly, _changed = regularize_instance_mask_ex(
        inst_mask, transform, min_area_m2=min_area_m2, min_thickness_m=min_thickness_m,
        simplify_tolerance_m=simplify_tolerance_m, min_keep_iou=min_keep_iou,
        min_edge_length_m=min_edge_length_m, diagonal_window_deg=diagonal_window_deg)
    return poly


# ---------------------------------------------------------------------------
# Step 2 & 3: dominant axis + angle snap + corner rebuild
# ---------------------------------------------------------------------------

def _dominant_angle_deg(poly: Polygon) -> float:
    coords = list(poly.exterior.coords)
    angles, weights = [], []
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)) % 90.0)
        weights.append(length)
    if not angles:
        return 0.0

    rad4 = np.radians(np.asarray(angles) * 4.0)
    w = np.asarray(weights)
    mean_rad4 = math.atan2(float(np.sum(w * np.sin(rad4))),
                            float(np.sum(w * np.cos(rad4))))
    return math.degrees(mean_rad4 / 4.0) % 90.0


def _snap_edge_angle(raw_deg: float, diagonal_window_deg: float) -> float:
    folded = raw_deg % 180.0
    targets = [0.0, 90.0, 180.0]
    if diagonal_window_deg > 0:
        targets += [45.0, 135.0]
    nearest = min(targets, key=lambda t: min(abs(folded - t), 180.0 - abs(folded - t)))
    if nearest in (45.0, 135.0):
        dist = min(abs(folded - nearest), 180.0 - abs(folded - nearest))
        if dist > diagonal_window_deg:
            nearest = 0.0 if min(abs(folded - 0.0), 180.0 - abs(folded - 0.0)) <= \
                min(abs(folded - 90.0), 180.0 - abs(folded - 90.0)) else 90.0
    return nearest % 180.0


def _line_intersection(p1, d1, p2, d2) -> Optional[tuple[float, float]]:
    x1, y1 = p1
    dx1, dy1 = d1
    x2, y2 = p2
    dx2, dy2 = d2
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return None
    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return (x1 + t * dx1, y1 + t * dy1)


def _same_direction(d1: tuple[float, float], d2: tuple[float, float], eps_deg: float = 1.0) -> bool:
    dot = abs(d1[0] * d2[0] + d1[1] * d2[1])
    return dot >= math.cos(math.radians(eps_deg))


def _snap_and_rebuild(
    poly: Polygon,
    dominant_angle_deg: float,
    min_edge_length: float,
    diagonal_window_deg: float,
) -> Optional[Polygon]:
    try:
        centroid = poly.centroid
        flat = shapely_rotate(poly, -dominant_angle_deg, origin=centroid, use_radians=False)
        coords = list(flat.exterior.coords)[:-1]
        if len(coords) < 3:
            return None

        edges = []
        for (x1, y1), (x2, y2) in zip(coords, coords[1:] + coords[:1]):
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            raw_deg = math.degrees(math.atan2(dy, dx))
            snapped_deg = _snap_edge_angle(raw_deg, diagonal_window_deg)
            rad = math.radians(snapped_deg)
            direction = (math.cos(rad), math.sin(rad))
            midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            edges.append([midpoint, direction, length])

        merged = []
        for mid, direction, length in edges:
            if merged and _same_direction(merged[-1][1], direction):
                prev_mid, prev_dir, prev_len = merged[-1]
                total = prev_len + length
                new_mid = (
                    (prev_mid[0] * prev_len + mid[0] * length) / total,
                    (prev_mid[1] * prev_len + mid[1] * length) / total,
                )
                merged[-1] = [new_mid, prev_dir, total]
            else:
                merged.append([mid, direction, length])

        if len(merged) > 1 and _same_direction(merged[-1][1], merged[0][1]):
            prev_mid, prev_dir, prev_len = merged[-1]
            first_mid, first_dir, first_len = merged[0]
            total = prev_len + first_len
            new_mid = (
                (prev_mid[0] * prev_len + first_mid[0] * first_len) / total,
                (prev_mid[1] * prev_len + first_mid[1] * first_len) / total,
            )
            merged[0] = [new_mid, first_dir, total]
            merged.pop()

        kept = [e for e in merged if e[2] >= min_edge_length]
        if len(kept) < 3:
            return None

        corners = []
        n = len(kept)
        for i in range(n):
            mid_a, dir_a, _ = kept[i - 1]
            mid_b, dir_b, _ = kept[i]
            point = _line_intersection(mid_a, dir_a, mid_b, dir_b)
            if point is None:
                point = ((mid_a[0] + mid_b[0]) / 2.0, (mid_a[1] + mid_b[1]) / 2.0)
            corners.append(point)

        rebuilt = Polygon(corners)
        if not rebuilt.is_valid:
            rebuilt = rebuilt.buffer(0)
        if rebuilt.is_empty or rebuilt.geom_type != "Polygon":
            return None

        result = shapely_rotate(rebuilt, dominant_angle_deg, origin=centroid, use_radians=False)
        return result if result.is_valid and not result.is_empty else None
    except Exception:
        return None


def regularize_polygon(
    poly: Polygon,
    *,
    min_keep_iou: float = DEFAULT_MIN_KEEP_IOU,
    min_edge_length: float = DEFAULT_MIN_EDGE_LENGTH,
    diagonal_window_deg: float = DEFAULT_DIAGONAL_WINDOW_DEG,
) -> tuple[Polygon, bool]:
    if poly is None or poly.is_empty or not poly.is_valid or poly.area <= 0:
        return poly, False

    angle = _dominant_angle_deg(poly)
    snapped = _snap_and_rebuild(poly, angle, min_edge_length, diagonal_window_deg)
    if snapped is None:
        return poly, False

    intersection = poly.intersection(snapped).area
    union = poly.union(snapped).area
    iou = (intersection / union) if union > 0 else 0.0
    if iou < min_keep_iou:
        return poly, False
    return snapped, True


def regularize_mask(
    mask: np.ndarray,
    *,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
    min_keep_iou: float = DEFAULT_MIN_KEEP_IOU,
    min_edge_length: float = DEFAULT_MIN_EDGE_LENGTH,
    diagonal_window_deg: float = DEFAULT_DIAGONAL_WINDOW_DEG,
) -> Optional[Polygon]:
    simplified = mask_to_polygon(mask, simplify_tolerance)
    if simplified is None:
        return None
    result, _ = regularize_polygon(
        simplified, min_keep_iou=min_keep_iou, min_edge_length=min_edge_length,
        diagonal_window_deg=diagonal_window_deg)
    return result

if __name__ == "__main__":
    demo = np.zeros((120, 120), dtype=np.uint8)
    demo[10:80, 10:60] = 1
    demo[50:110, 10:100] = 1
    poly = regularize_mask(demo)
    if poly is None:
        print("No polygon produced — check opencv-python and shapely are installed.")
    else:
        print(f"Vertices: {len(poly.exterior.coords) - 1}, area: {poly.area:.1f}")