"""Maske/poligon islemleri, kutu birlestirme (NMS), dilim-sinirlarini
birlestirme ve görselleştirme (kutu çizimi) yardımcı fonksiyonları."""
import numpy as np
import cv2


def is_narrow_sliver(poly, min_width_m=2.2):
    """Poligonun en dar kenari belirtilen metreden kucukse True doner."""
    if poly is None or poly.is_empty:
        return True
    mrr = poly.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return True
    coords = list(mrr.exterior.coords)
    edge_lengths = [
        ((coords[i][0] - coords[i+1][0])**2 + (coords[i][1] - coords[i+1][1])**2)**0.5
        for i in range(len(coords) - 1)
    ]
    if not edge_lengths:
        return True
    return min(edge_lengths) < min_width_m


def merge_seam_polygons(polygons, seam_tol_m=1.0, min_overlap_ratio=0.30):
    """
    Sadece belirgin şekilde üst üste binen parçaları birleştirir.
    Yan yana veya alt alta duran binaların yapışmasını engeller.
    """
    if not polygons:
        return []

    merged = []
    polys = list(polygons)

    while polys:
        current = polys.pop(0)
        did_merge = False

        for i, other in enumerate(polys):
            if current.intersects(other):
                inter = current.intersection(other)
                inter_area = inter.area
                min_area = min(current.area, other.area)

                # 1. ŞART: Kesişim alanı küçük parçanın en az %30'u olmalı (0.15 yetersiz kalıyor)
                if min_area > 0 and (inter_area / min_area) >= min_overlap_ratio:
                    union_poly = current.union(other)
                    if union_poly.geom_type == "Polygon" and union_poly.is_valid:
                        # 2. ŞART: İki bina birleşince doluluk oranı ciddi düşer (Z/L/basamak şekli alır).
                        # Aynı binanın iki yarısı ise dikdörtgen kalır (oran > 0.80).
                        if union_poly.area / union_poly.convex_hull.area > 0.80:
                            polys.pop(i)
                            polys.append(union_poly)
                            did_merge = True
                            break

        if not did_merge:
            merged.append(current)

    return merged


def apply_nms(boxes_meta, iou_threshold=0.45):
    """
    boxes_meta: [([x1, y1, x2, y2], score, source), ...]
    Dönen: Filtrelenmiş boxes_meta listesi
    """
    if not boxes_meta:
        return []
    boxes_meta_sorted = sorted(boxes_meta, key=lambda x: x[1], reverse=True)
    keep = []
    while boxes_meta_sorted:
        current = boxes_meta_sorted.pop(0)
        keep.append(current)

        curr_box = current[0]
        c_x1, c_y1, c_x2, c_y2 = curr_box
        curr_area = (c_x2 - c_x1) * (c_y2 - c_y1)

        remaining = []
        for item in boxes_meta_sorted:
            box_i = item[0]
            b_x1, b_y1, b_x2, b_y2 = box_i
            box_area = (b_x2 - b_x1) * (b_y2 - b_y1)
            inter_x1 = max(c_x1, b_x1)
            inter_y1 = max(c_y1, b_y1)
            inter_x2 = min(c_x2, b_x2)
            inter_y2 = min(c_y2, b_y2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            union_area = curr_area + box_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0

            if iou < iou_threshold:
                remaining.append(item)
        boxes_meta_sorted = remaining
    return keep


def mask_bbox_and_area(mask):
    ys = np.any(mask, axis=1)
    if not ys.any():
        return None, 0
    xs = np.any(mask, axis=0)
    y_idx = np.where(ys)[0]
    x_idx = np.where(xs)[0]
    y1, y2 = int(y_idx[0]), int(y_idx[-1]) + 1
    x1, x2 = int(x_idx[0]), int(x_idx[-1]) + 1
    area = int(mask.sum())
    return (y1, y2, x1, x2), area


def crop_mask_to_bbox(mask, bbox):
    y1, y2, x1, x2 = bbox
    return np.ascontiguousarray(mask[y1:y2, x1:x2])


def fast_overlap_ratio(mask_a, bbox_a, area_a, mask_b, bbox_b, area_b):
    if bbox_a is None or bbox_b is None:
        return 0.0
    ay1, ay2, ax1, ax2 = bbox_a
    by1, by2, bx1, bx2 = bbox_b
    iy1, iy2 = max(ay1, by1), min(ay2, by2)
    ix1, ix2 = max(ax1, bx1), min(ax2, bx2)
    if iy1 >= iy2 or ix1 >= ix2:
        return 0.0
    sub_a = mask_a[iy1 - ay1:iy2 - ay1, ix1 - ax1:ix2 - ax1]
    sub_b = mask_b[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1]
    intersection = int(np.logical_and(sub_a, sub_b).sum())
    if intersection == 0:
        return 0.0
    smaller_area = min(area_a, area_b)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def remove_thin_necks_and_tails(poly, dist_m=0.8):
    if poly is None or poly.is_empty:
        return poly
    eroded = poly.buffer(-dist_m)
    if eroded.is_empty:
        return poly
    return eroded.buffer(dist_m * 0.8)


def compute_tile_seam_lines(h, w, tile_size, overlap_ratio):
    if h <= tile_size and w <= tile_size:
        return [0, h], [0, w]
    step = max(1, int(tile_size * (1 - overlap_ratio)))
    y_starts = sorted(set(list(range(0, max(h - tile_size, 0) + 1, step)) + [max(h - tile_size, 0)]))
    x_starts = sorted(set(list(range(0, max(w - tile_size, 0) + 1, step)) + [max(w - tile_size, 0)]))
    y_seams = sorted({y1 for y1 in y_starts} | {min(y1 + tile_size, h) for y1 in y_starts})
    x_seams = sorted({x1 for x1 in x_starts} | {min(x1 + tile_size, w) for x1 in x_starts})
    return y_seams, x_seams


def bboxes_touch_near_seam(bbox_a, bbox_b, y_seams, x_seams, touch_tol_px=6, seam_tol_px=20):
    ay1, ay2, ax1, ax2 = bbox_a
    by1, by2, bx1, bx2 = bbox_b

    y_overlap = min(ay2, by2) - max(ay1, by1)
    if y_overlap > 0:
        gap = bx1 - ax2
        if 0 <= gap <= touch_tol_px and any(abs(ax2 - s) <= seam_tol_px for s in x_seams):
            return True
        gap = ax1 - bx2
        if 0 <= gap <= touch_tol_px and any(abs(ax1 - s) <= seam_tol_px for s in x_seams):
            return True

    x_overlap = min(ax2, bx2) - max(ax1, bx1)
    if x_overlap > 0:
        gap = by1 - ay2
        if 0 <= gap <= touch_tol_px and any(abs(ay2 - s) <= seam_tol_px for s in y_seams):
            return True
        gap = ay1 - by2
        if 0 <= gap <= touch_tol_px and any(abs(ay1 - s) <= seam_tol_px for s in y_seams):
            return True

    return False


def draw_orig_box_on_image(img_array, box_pixel_bounds, color=(51, 136, 255), thickness=3):
    if box_pixel_bounds is None:
        if img_array.ndim == 2:
            return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        return img_array.copy()

    if img_array.ndim == 2:
        display_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        display_img = img_array.copy()

    h, w = display_img.shape[:2]
    x1, y1, x2, y2 = box_pixel_bounds
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)
    return display_img

