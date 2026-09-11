"""Haritadan bagimsiz "TIF Dosyasindan Tespit" bolumu: kullanicinin kendi
GeoTIFF'ini yuklemesi, bozuk/eksik coğrafi referansi otomatik onarma akisi,
kendi mini haritasinda alan secimi ve tespiti calistirma.
"""
import os

import folium
import streamlit as st
from shapely.geometry import box
from streamlit_folium import st_folium
from folium.plugins import Draw
from folium.raster_layers import ImageOverlay

from .config import ZOOM_LEVEL
from .detection import run_detection_pipeline
from .geo_utils import zoom_to_gsd_m
from .tif_upload import (
    build_tif_preview_array,
    crop_uploaded_tif_to_aoi,
    get_tif_bounds_4326,
    repair_missing_georeference,
)


def render_tif_section(settings, yolo_model, sam):
    """Haritadan bagimsiz TIF yukleme + alan secimi + tespit bolumunu cizer."""
    # --- 1B / 2B. TIF DOSYASINDAN TESPİT (HARİTADAN AYRI) ---
    st.divider()
    st.subheader("📁 TIF Dosyasından Tespit")
    st.caption(
        "Haritadan bağımsız olarak kendi .tif/.tiff uydu veya hava görüntünüzü yükleyin, "
        "aşağıdaki mini haritada dikdörtgen aracıyla bir alan seçin ve aynı YOLO+SAM3 "
        "boru hattıyla (maskeleme, gölge filtreleme, geometri iyileştirme, görselleştirme) tespiti çalıştırın."
    )

    for _k, _v in {
        "tif_uploaded_name": None, "tif_uploaded_path": None, "tif_bounds_4326": None,
        "tif_preview_arr": None, "tif_map_center": None, "tif_selected_box": None,
        "tif_needs_reference": False, "tif_broken_path": None, "tif_broken_name": None,
    }.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    uploaded_tif = st.file_uploader("GeoTIFF yükleyin (.tif / .tiff)", type=["tif", "tiff"], key="tif_uploader")

    if uploaded_tif is not None:
        if st.session_state["tif_uploaded_name"] != uploaded_tif.name and st.session_state["tif_broken_name"] != uploaded_tif.name:
            tif_uploads_dir = os.path.join(os.getcwd(), "streamlit_temp", "tif_uploads")
            os.makedirs(tif_uploads_dir, exist_ok=True)
            saved_tif_path = os.path.join(tif_uploads_dir, uploaded_tif.name)
            with open(saved_tif_path, "wb") as f:
                f.write(uploaded_tif.getbuffer())

            st.session_state["tif_needs_reference"] = False
            st.session_state["tif_broken_path"] = None
            st.session_state["tif_broken_name"] = None

            try:
                _tif_bounds_4326 = get_tif_bounds_4326(saved_tif_path)
                _preview_arr = build_tif_preview_array(saved_tif_path)
                st.session_state["tif_uploaded_name"] = uploaded_tif.name
                st.session_state["tif_uploaded_path"] = saved_tif_path
                st.session_state["tif_bounds_4326"] = _tif_bounds_4326
                st.session_state["tif_preview_arr"] = _preview_arr
                st.session_state["tif_map_center"] = [
                    (_tif_bounds_4326[1] + _tif_bounds_4326[3]) / 2,
                    (_tif_bounds_4326[0] + _tif_bounds_4326[2]) / 2,
                ]
                st.session_state["tif_selected_box"] = None
            except Exception as e:
                st.session_state["tif_uploaded_name"] = None
                st.session_state["tif_uploaded_path"] = None
                st.session_state["tif_needs_reference"] = True
                st.session_state["tif_broken_path"] = saved_tif_path
                st.session_state["tif_broken_name"] = uploaded_tif.name
                st.warning(
                    f"⚠️ '{uploaded_tif.name}' dosyasının coğrafi referansı okunamadı: {e}\n\n"
                    "Bu genellikle bir çözünürlük artırma / süper-çözünürlük işleminden sonra olur. "
                    "Aşağıya bu görüntünün **işlem görmemiş, coğrafi referansı sağlam olan orijinalini** "
                    "yükleyin - referansı oradan alıp otomatik olarak onaracağım, tekrar bir şey "
                    "indirmenize gerek kalmayacak."
                )

    if st.session_state.get("tif_needs_reference") and st.session_state.get("tif_broken_path"):
        st.markdown(
            f"**Referans TIF yükleyin** — `{st.session_state.get('tif_broken_name', '')}` "
            "dosyasının coğrafi referansını buradan alacağım:"
        )
        ref_tif = st.file_uploader(
            "Orijinal (coğrafi referansı sağlam) TIF", type=["tif", "tiff"], key="tif_reference_uploader"
        )
        if ref_tif is not None:
            ref_dir = os.path.join(os.getcwd(), "streamlit_temp", "tif_uploads")
            os.makedirs(ref_dir, exist_ok=True)
            ref_path = os.path.join(ref_dir, "ref_" + ref_tif.name)
            with open(ref_path, "wb") as f:
                f.write(ref_tif.getbuffer())

            repaired_path = os.path.join(ref_dir, "georef_" + st.session_state["tif_broken_name"])
            with st.spinner("Coğrafi referans onarılıyor..."):
                try:
                    repair_missing_georeference(ref_path, st.session_state["tif_broken_path"], repaired_path)
                    _tif_bounds_4326 = get_tif_bounds_4326(repaired_path)
                    _preview_arr = build_tif_preview_array(repaired_path)
                    st.session_state["tif_uploaded_name"] = st.session_state["tif_broken_name"]
                    st.session_state["tif_uploaded_path"] = repaired_path
                    st.session_state["tif_bounds_4326"] = _tif_bounds_4326
                    st.session_state["tif_preview_arr"] = _preview_arr
                    st.session_state["tif_map_center"] = [
                        (_tif_bounds_4326[1] + _tif_bounds_4326[3]) / 2,
                        (_tif_bounds_4326[0] + _tif_bounds_4326[2]) / 2,
                    ]
                    st.session_state["tif_selected_box"] = None
                    st.session_state["tif_needs_reference"] = False
                    st.session_state["tif_broken_path"] = None
                    st.success("✅ Coğrafi referans onarıldı. Aşağıda alan seçimine geçebilirsiniz.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Referans onarma hatası: {e}")

    tif_uploaded_path = st.session_state.get("tif_uploaded_path")
    tif_bounds_4326 = st.session_state.get("tif_bounds_4326")

    if tif_uploaded_path and tif_bounds_4326:
        st.markdown("**Yüklenen TIF üzerinde analiz alanı seçin:**")

        m_tif = folium.Map(location=st.session_state["tif_map_center"], zoom_start=17, max_zoom=22)

        tb_minx, tb_miny, tb_maxx, tb_maxy = tif_bounds_4326
        ImageOverlay(
            image=st.session_state["tif_preview_arr"],
            bounds=[[tb_miny, tb_minx], [tb_maxy, tb_maxx]],
            opacity=1.0,
            name="Yüklenen TIF",
        ).add_to(m_tif)

        if st.session_state["tif_selected_box"] is not None:
            _t_minx, _t_miny, _t_maxx, _t_maxy = st.session_state["tif_selected_box"].bounds
            folium.Rectangle(
                bounds=[[_t_miny, _t_minx], [_t_maxy, _t_maxx]],
                color="#ff8c00",
                weight=3,
                fill=True,
                fill_opacity=0.08,
                tooltip="Seçilen Alan (TIF)",
            ).add_to(m_tif)

        draw_tif = Draw(
            export=False,
            draw_options={
                'polyline': False, 'polygon': False, 'circle': False,
                'marker': False, 'circlemarker': False, 'rectangle': True
            }
        )
        draw_tif.add_to(m_tif)

        tif_map_data = st_folium(m_tif, width="100%", height=550, key="tif_map")

        if tif_map_data and tif_map_data.get("all_drawings"):
            tif_draw_features = tif_map_data["all_drawings"]
            if len(tif_draw_features) > 0:
                t_coords = tif_draw_features[-1]["geometry"]["coordinates"][0]
                t_lons = [pt[0] for pt in t_coords]
                t_lats = [pt[1] for pt in t_coords]
                st.session_state["tif_selected_box"] = box(min(t_lons), min(t_lats), max(t_lons), max(t_lats))

        tif_selected_box = st.session_state["tif_selected_box"]

        st.markdown("**TIF üzerinde tespiti çalıştırın:**")
        if tif_selected_box is None:
            st.info("⚠️ Yukarıdaki mini haritada dikdörtgen aracıyla bir alan çizin.")
        elif yolo_model is None:
            st.error("YOLO modeli yüklenemedi.")
        else:
            tif_analyze_btn = st.button(
                "TIF Üzerinde Bina Çatılarını Tespit Et", type="primary", key="tif_analyze_btn"
            )

            if tif_analyze_btn:
                t_orig_minx, t_orig_miny, t_orig_maxx, t_orig_maxy = tif_selected_box.bounds

                t_buf_x = (t_orig_maxx - t_orig_minx) * (settings.edge_buffer_pct / 100.0)
                t_buf_y = (t_orig_maxy - t_orig_miny) * (settings.edge_buffer_pct / 100.0)
                t_minx, t_miny, t_maxx, t_maxy = (
                    t_orig_minx - t_buf_x, t_orig_miny - t_buf_y,
                    t_orig_maxx + t_buf_x, t_orig_maxy + t_buf_y,
                )

                tif_work_dir = os.path.join(os.getcwd(), "streamlit_temp_tif")
                os.makedirs(tif_work_dir, exist_ok=True)
                tif_aoi_path = os.path.join(tif_work_dir, "aoi_from_upload.tif")

                with st.spinner("Yüklenen TIF'ten seçilen alan kırpılıyor ve hazırlanıyor..."):
                    try:
                        crop_uploaded_tif_to_aoi(
                            tif_uploaded_path, [t_minx, t_miny, t_maxx, t_maxy], tif_aoi_path,
                            target_gsd_m=zoom_to_gsd_m(ZOOM_LEVEL),
                        )
                    except Exception as e:
                        st.error(f"TIF kırpma hatası: {e}")
                        st.stop()

                run_detection_pipeline(
                    tif_aoi_path, tif_work_dir, t_orig_minx, t_orig_miny, t_orig_maxx, t_orig_maxy,
                    settings, yolo_model, sam,
                )
                st.success("TIF üzerinde tespit tamamlandı!")

