"""Web Mercator tile matematigi, canli uydu goruntusu indirme ve
coğrafi kutu <-> piksel donusumleri."""
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import rasterio
import requests
import streamlit as st
from PIL import Image
from rasterio.transform import from_bounds

from .config import EARTH_RADIUS, ORIGIN_SHIFT, TILE_SIZE


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_bounds_3857(xtile, ytile, zoom):
    resolution = (2 * math.pi * EARTH_RADIUS / TILE_SIZE) / (2 ** zoom)
    minx = xtile * TILE_SIZE * resolution - ORIGIN_SHIFT
    maxx = (xtile + 1) * TILE_SIZE * resolution - ORIGIN_SHIFT
    maxy = ORIGIN_SHIFT - ytile * TILE_SIZE * resolution
    miny = ORIGIN_SHIFT - (ytile + 1) * TILE_SIZE * resolution
    return minx, miny, maxx, maxy


def zoom_to_gsd_m(zoom):
    """Web Mercator (EPSG:3857) zoom seviyesine karsilik gelen ekvatordaki GSD (m/piksel).
    tile_bounds_3857() icindeki cozunurluk hesabiyla ayni formul."""
    return (2 * math.pi * EARTH_RADIUS / TILE_SIZE) / (2 ** zoom)


def get_google_session_token(api_key):
    cached = st.session_state.get("google_session_token")
    if cached and cached.get("expiry", 0) > time.time() + 60:
        return cached["session"]

    resp = requests.post(
        f"https://tile.googleapis.com/v1/createSession?key={api_key}",
        json={"mapType": "satellite", "language": "tr-TR", "region": "TR"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    expiry = float(data.get("expiry", time.time() + 3600))
    st.session_state["google_session_token"] = {"session": data["session"], "expiry": expiry}
    return data["session"]


def build_tile_url(source, xt, yt, zoom, google_session=None, api_key=None):
    if source == "google":
        return f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{xt}/{yt}?session={google_session}&key={api_key}"
    return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{yt}/{xt}"


def _fetch_tile(session, xt, yt, zoom, source="esri", google_session=None, api_key=None):
    tile_url = build_tile_url(source, xt, yt, zoom, google_session=google_session, api_key=api_key)
    resp = session.get(tile_url, timeout=30)
    resp.raise_for_status()
    tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return xt, yt, tile_img


def download_satellite_geotiff(bbox, output_path, zoom=19, max_workers=8, source="esri", api_key=None):
    minx, miny, maxx, maxy = bbox
    x_tl, y_tl = deg2num(maxy, minx, zoom)
    x_br, y_br = deg2num(miny, maxx, zoom)

    x_tiles = list(range(min(x_tl, x_br), max(x_tl, x_br) + 1))
    y_tiles = list(range(min(y_tl, y_br), max(y_tl, y_br) + 1))

    if len(x_tiles) * len(y_tiles) > 400:
        raise RuntimeError(f"Seçilen alan çok büyük ({len(x_tiles)}x{len(y_tiles)} tile).")

    google_session = None
    if source == "google":
        if not api_key:
            raise RuntimeError("Google Maps API Key bulunamadı.")
        google_session = get_google_session_token(api_key)

    stitched = Image.new("RGB", (len(x_tiles) * TILE_SIZE, len(y_tiles) * TILE_SIZE))
    x_index = {xt: ix for ix, xt in enumerate(x_tiles)}
    y_index = {yt: iy for iy, yt in enumerate(y_tiles)}

    errors = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_tile, session, xt, yt, zoom, source=source, google_session=google_session, api_key=api_key)
                for xt in x_tiles for yt in y_tiles
            ]
            for future in as_completed(futures):
                try:
                    xt, yt, tile_img = future.result()
                    stitched.paste(tile_img, (x_index[xt] * TILE_SIZE, y_index[yt] * TILE_SIZE))
                except Exception as e:
                    errors.append(str(e))

    if errors:
        raise RuntimeError(f"{len(errors)} tile indirilemedi.")

    arr = np.array(stitched)
    left_3857, _, _, top_3857 = tile_bounds_3857(x_tiles[0], y_tiles[0], zoom)
    _, bottom_3857, right_3857, _ = tile_bounds_3857(x_tiles[-1], y_tiles[-1], zoom)

    transform = from_bounds(left_3857, bottom_3857, right_3857, top_3857, arr.shape[1], arr.shape[0])

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=3,
        dtype=arr.dtype, crs="EPSG:3857", transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(arr[:, :, band], band + 1)

    return output_path


def get_orig_box_pixel_bounds(orig_box_4326, transform, crs):
    """Bir EPSG:4326 kutusunun, verilen transform/crs'e göre piksel sınırlarını döner."""
    try:
        gdf_box = gpd.GeoDataFrame(geometry=[orig_box_4326], crs="EPSG:4326").to_crs(crs)
        minx, miny, maxx, maxy = gdf_box.total_bounds
        inv_transform = ~transform
        col_a, row_a = inv_transform * (minx, miny)
        col_b, row_b = inv_transform * (maxx, maxy)
        x1, x2 = sorted([col_a, col_b])
        y1, y2 = sorted([row_a, row_b])
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None
