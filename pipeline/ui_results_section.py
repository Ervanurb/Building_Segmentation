"""Tespit sonrasi gosterilen bolumler: Ara Asama Gorselleri (YOLO kutulari,
ham/temiz maskeler), Tespit Sonucu (harita uzerinde poligonlar) ve YOLO
egitim metrikleri paneli. Tespitin harita ya da TIF akisindan gelmis olmasina
bakmaksizin (st.session_state uzerinden) ayni sekilde calisir.
"""
import cv2
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from .config import OPACITY, ZOOM_LEVEL
from .geo_utils import get_google_session_token, get_orig_box_pixel_bounds
from .mask_utils import draw_orig_box_on_image
from .models import load_yolo_metrics


def render_results_section(settings, active_source, google_maps_api_key, mask_color):
    """Ara Asama Gorselleri + Tespit Sonucu + Model Metrikleri panelini cizer."""
    # --- 2.5. ARA AŞAMA GÖRSELLERİ ---
    if st.session_state.get("debug_img_rgb") is not None:
        st.divider()
        st.subheader("🔍 Ara Aşama Görselleri")

        debug_img_rgb = st.session_state["debug_img_rgb"]
        debug_boxes = st.session_state.get("debug_boxes") or []
        n_yolo = st.session_state.get("debug_boxes_yolo_count") or 0

        _orig_box_pixel_bounds = None
        if (st.session_state.get("result_orig_box") is not None
                and st.session_state.get("result_transform") is not None
                and st.session_state.get("result_crs") is not None):
            _orig_box_pixel_bounds = get_orig_box_pixel_bounds(
                st.session_state["result_orig_box"],
                st.session_state["result_transform"],
                st.session_state["result_crs"],
            )

        tab_boxes, tab_masks = st.tabs(["1️⃣ YOLO Kutuları", "2️⃣ Ham Maske Dağılımı"])

        with tab_boxes:
            debug_img_boxes = debug_img_rgb.copy()
            for i, bx in enumerate(debug_boxes):
                x1, y1, x2, y2 = [int(v) for v in bx]
                cv2.rectangle(debug_img_boxes, (x1, y1), (x2, y2), (255, 0, 0), 3)
            debug_img_boxes = draw_orig_box_on_image(debug_img_boxes, _orig_box_pixel_bounds)
            st.image(debug_img_boxes, caption=f"Kırmızı: YOLO ({n_yolo}) | Mavi: Seçilen alan", width='stretch')
            if settings.add_text_prompt_search:
                st.caption("Not: SAM3 metin prompt taraması burada gösterilmez (box üretmez), "
                           "ama sonuç maskesine dahildir. Toplam eklenen maske sayısı sidebar'da görünür.")

        with tab_masks:
            raw_mask = st.session_state.get("result_mask_raw")
            mask_pre_shadow = st.session_state.get("result_mask_pre_shadow")
            shadow_mask = st.session_state.get("result_shadow_mask")
            removed_ratio = st.session_state.get("shadow_removed_ratio")

            if raw_mask is None:
                st.info("Maske mevcut değil.")
            elif shadow_mask is not None:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    pre_display = draw_orig_box_on_image(
                        (mask_pre_shadow * 255).astype(np.uint8), _orig_box_pixel_bounds
                    )
                    st.image(pre_display, caption="Ham SAM3 Maskesi (gölgeden önce)", width='stretch')
                with col_b:
                    shadow_display = draw_orig_box_on_image(
                        (shadow_mask * 255).astype(np.uint8), _orig_box_pixel_bounds
                    )
                    st.image(shadow_display, caption="Gölge Maskesi", width='stretch')
                with col_c:
                    final_display = draw_orig_box_on_image(
                        (raw_mask * 255).astype(np.uint8), _orig_box_pixel_bounds
                    )
                    st.image(final_display, caption="Temiz Maske (gölge çıkarılmış)", width='stretch')
                if removed_ratio is not None:
                    st.write(f"Gölge nedeniyle çıkarılan alan: %{removed_ratio * 100:.2f}")
            else:
                raw_display = draw_orig_box_on_image((raw_mask * 255).astype(np.uint8), _orig_box_pixel_bounds)
                st.image(raw_display, caption="SAM Instance Birleşimi (Ham Referans)", width='stretch')


    # --- 3. SONUÇ HARİTASI ---
    if st.session_state.get("result_gdf") is not None and st.session_state.get("result_orig_box") is not None:
        st.divider()
        st.subheader("3. Tespit Sonucu")

        gdf_results = st.session_state["result_gdf"]
        minx, miny, maxx, maxy = st.session_state["result_orig_box"].bounds

        m_res = folium.Map(
            location=[(miny + maxy) / 2, (minx + maxx) / 2],
            zoom_start=ZOOM_LEVEL,
            max_zoom=22,
            tiles=None
        )

        if active_source == "google" and google_maps_api_key:
            try:
                _g_session_res = get_google_session_token(google_maps_api_key)
                folium.TileLayer(
                    tiles=f"https://tile.googleapis.com/v1/2dtiles/{{z}}/{{x}}/{{y}}?session={_g_session_res}&key={google_maps_api_key}",
                    attr="Google Maps",
                    max_zoom=22,
                    max_native_zoom=20
                ).add_to(m_res)
            except Exception:
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri World Imagery",
                    max_zoom=22,
                    max_native_zoom=19
                ).add_to(m_res)
        else:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                max_zoom=22,
                max_native_zoom=19
            ).add_to(m_res)

        folium.GeoJson(
            gdf_results,
            style_function=lambda x: {
                'fillColor': mask_color, 'color': mask_color, 'weight': 2, 'fillOpacity': OPACITY
            }
        ).add_to(m_res)

        st_folium(m_res, width="100%", height=550, key="result_map")

        if st.session_state.get("shadow_removed_ratio") is not None:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Tespit Edilen Bağımsız Bina Çatısı Sayısı", len(gdf_results))
            with col_m2:
                st.metric("Gölge Nedeniyle Çıkarılan Alan", f"%{st.session_state['shadow_removed_ratio'] * 100:.2f}")
        else:
            st.metric("Tespit Edilen Bağımsız Bina Çatısı Sayısı", len(gdf_results))

