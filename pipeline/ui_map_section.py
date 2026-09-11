"""Canli harita uzerinde alan secimi ve "Bu Alanda Bina Catilarini Tespit Et"
akisi. render_map_section() cagrildiginda haritayi cizer, secimi st.session_state
icinde tutar ve tespit calistirildiysa run_detection_pipeline()'i cagirir.
Doner: active_source ("google" ya da "esri") - sonuc haritasinda ayni katmani
kullanabilmek icin ui_results_section.py'a geçirilir.
"""
import os

import folium
import streamlit as st
from shapely.geometry import box
from streamlit_folium import st_folium
from folium.plugins import Draw

from .config import PRESET_LOCATIONS, ZOOM_LEVEL
from .detection import run_detection_pipeline
from .geo_utils import download_satellite_geotiff, get_google_session_token


def render_map_section(settings, map_source, google_maps_api_key, yolo_model, sam):
    """Harita + alan secimi + tespit butonu bolumunu cizer.
    Doner: active_source ("google" ya da "esri")."""

    # --- SESSION STATE ---
    if "selected_box" not in st.session_state:
        st.session_state["selected_box"] = None
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = PRESET_LOCATIONS["Muratpaşa(Antalya)"]
    if "result_gdf" not in st.session_state:
        st.session_state["result_gdf"] = None
    if "result_orig_box" not in st.session_state:
        st.session_state["result_orig_box"] = None
    if "result_mask_raw" not in st.session_state:
        st.session_state["result_mask_raw"] = None
    if "result_mask_pre_shadow" not in st.session_state:
        st.session_state["result_mask_pre_shadow"] = None
    if "result_shadow_mask" not in st.session_state:
        st.session_state["result_shadow_mask"] = None
    if "shadow_removed_ratio" not in st.session_state:
        st.session_state["shadow_removed_ratio"] = None
    if "debug_img_rgb" not in st.session_state:
        st.session_state["debug_img_rgb"] = None
    if "debug_boxes" not in st.session_state:
        st.session_state["debug_boxes"] = None
    if "debug_boxes_yolo_count" not in st.session_state:
        st.session_state["debug_boxes_yolo_count"] = None

    # --- 1. HARİTA + ALAN SEÇİMİ ---
    st.subheader("1. Analiz Edilecek Alanı Seçin")

    col_preset, col_goto = st.columns([3, 1])
    with col_preset:
        preset_choice = st.selectbox("Hazır Konuma Git", list(PRESET_LOCATIONS.keys()))
    with col_goto:
        st.write("")
        st.write("")
        if st.button("Git"):
            st.session_state["map_center"] = PRESET_LOCATIONS[preset_choice]
            st.rerun()

    col_coord, col_coord_goto = st.columns([3, 1])
    with col_coord:
        coord_input = st.text_input(
            "Koordinat Gir (Enlem, Boylam)",
            placeholder="ör. 39.9334, 32.8597",
            help="Google Maps veya başka bir haritadan kopyaladığın 'enlem, boylam' "
                 "çiftini virgülle ayırarak buraya yapıştırabilirsin."
        )
    with col_coord_goto:
        st.write("")
        st.write("")
        if st.button("Git", key="coord_goto_btn"):
            try:
                lat_str, lon_str = coord_input.split(",")
                lat, lon = float(lat_str.strip()), float(lon_str.strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    st.session_state["map_center"] = [lat, lon]
                    st.rerun()
                else:
                    st.sidebar.error("Enlem -90/90, boylam -180/180 aralığında olmalı.")
            except (ValueError, AttributeError):
                st.sidebar.error("Koordinat formatı hatalı - 'enlem, boylam' şeklinde gir (ör. 39.9334, 32.8597).")

    active_source = "google" if map_source.startswith("Google") else "esri"
    m = folium.Map(location=st.session_state["map_center"], zoom_start=15, max_zoom=22)

    if active_source == "google":
        try:
            _g_session = get_google_session_token(google_maps_api_key)
            folium.TileLayer(
                tiles=f"https://tile.googleapis.com/v1/2dtiles/{{z}}/{{x}}/{{y}}?session={_g_session}&key={google_maps_api_key}",
                attr="Google Maps",
                name="Uydu Görüntüsü (Google)",
                overlay=False,
                control=True,
                max_zoom=22,
                max_native_zoom=20,
            ).add_to(m)
        except Exception as e:
            st.sidebar.error(f"Google token hatası: {e}. Esri'ye dönülüyor.")
            active_source = "esri"

    if active_source == "esri":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Uydu Görüntüsü",
            overlay=False,
            control=True,
            max_zoom=22,
            max_native_zoom=19,
        ).add_to(m)

    if st.session_state["selected_box"] is not None:
        _sel_minx, _sel_miny, _sel_maxx, _sel_maxy = st.session_state["selected_box"].bounds
        folium.Rectangle(
            bounds=[[_sel_miny, _sel_minx], [_sel_maxy, _sel_maxx]],
            color="#3388ff",
            weight=3,
            fill=True,
            fill_opacity=0.08,
            tooltip="Seçilen Alan",
        ).add_to(m)

    draw = Draw(
        export=False,
        draw_options={
            'polyline': False, 'polygon': False, 'circle': False,
            'marker': False, 'circlemarker': False, 'rectangle': True
        }
    )
    draw.add_to(m)

    map_data = st_folium(m, width="100%", height=550, key="main_map")

    if map_data and map_data.get("all_drawings"):
        draw_features = map_data["all_drawings"]
        if len(draw_features) > 0:
            coords = draw_features[-1]["geometry"]["coordinates"][0]
            lons = [pt[0] for pt in coords]
            lats = [pt[1] for pt in coords]
            st.session_state["selected_box"] = box(min(lons), min(lats), max(lons), max(lats))
            st.session_state["map_center"] = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]

    selected_box = st.session_state["selected_box"]

    # --- 2. TESPİT BUTONU ---
    st.divider()
    st.subheader("2. Tespiti Çalıştır")

    if selected_box is None:
        st.info("⚠️ Önce haritada dikdörtgen aracıyla bir alan çizin.")
    elif yolo_model is None:
        st.error("YOLO modeli yüklenemedi.")
    else:
        analyze_btn = st.button("Bu Alanda Bina Çatılarını Tespit Et", type="primary")

        if analyze_btn:
            orig_minx, orig_miny, orig_maxx, orig_maxy = selected_box.bounds

            buf_x = (orig_maxx - orig_minx) * (settings.edge_buffer_pct / 100.0)
            buf_y = (orig_maxy - orig_miny) * (settings.edge_buffer_pct / 100.0)
            minx, miny, maxx, maxy = orig_minx - buf_x, orig_miny - buf_y, orig_maxx + buf_x, orig_maxy + buf_y

            work_dir = os.path.join(os.getcwd(), "streamlit_temp")
            os.makedirs(work_dir, exist_ok=True)
            tif_path = os.path.join(work_dir, "aoi.tif")

            with st.spinner(f"Uydu görüntüsü indiriliyor ({active_source})..."):
                try:
                    download_satellite_geotiff(
                        [minx, miny, maxx, maxy], tif_path, zoom=ZOOM_LEVEL,
                        source=active_source, api_key=google_maps_api_key,
                    )
                except Exception as e:
                    st.error(f"Görüntü indirme hatası: {e}")
                    st.stop()

            run_detection_pipeline(tif_path, work_dir, orig_minx, orig_miny, orig_maxx, orig_maxy, settings, yolo_model, sam)
            st.success("Tespit tamamlandı!")


    return active_source
