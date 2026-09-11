"""Sol kenar cubugundaki tum ayar widget'lari. render_sidebar(), sidebar'i
cizer ve bir DetectionSettings nesnesi + harita kaynagi + maske rengi doner.
"""
import os

import streamlit as st

from .config import DetectionSettings


def render_sidebar(google_maps_api_key):
    """Sidebar'i cizer. Doner: (settings, map_source, mask_color)."""

    # --- SIDEBAR ---
    st.sidebar.header("Model & Parametreler")

    nms_threshold = st.sidebar.slider("NMS IoU Eşik Değeri", 0.1, 0.9, 0.45, 0.05, help="Çift maskeleri engellemek için kutu çakışma eşiği.")

    edge_buffer_pct = st.sidebar.slider(
        "Kenar Payı (%)", 0, 50, 20, 5,
        help="Seçilen alanın kenarında yarım kalan binaları tamamlamak için indirme alanını genişletir."
    )

    st.sidebar.header("Harita Kaynağı")
    map_source_options = []
    if google_maps_api_key:
        map_source_options.append("Google Satellite (Maps Tile API)")
    map_source_options.append("Esri World Imagery")
    map_source = st.sidebar.radio("Uydu Görüntüsü Kaynağı", map_source_options)

    if not google_maps_api_key:
        st.sidebar.warning("google_maps_api_key bulunamadı (.env dosyasını kontrol edin). Sadece Esri kullanılabilir.")

    st.sidebar.header("Gölge Filtreleme")
    st.sidebar.caption(
        "Basit parlaklık eşiği her tespitte otomatik uygulanır (kapatılamaz). "
        "İstersen buna ek olarak bir model-tabanlı yöntem de seçebilirsin."
    )
    extra_shadow_method = st.sidebar.radio(
        "Ek Gölge Tespit Yöntemi (isteğe bağlı)",
        ["Yok", "Shadow U-Net (eğitilmiş model)", "SAM3 (zero-shot metin prompt)"],
        help="Parlaklık eşiğine ek olarak ikinci bir yöntemle de gölge tespiti yapar; "
             "ikisinin bulduğu alanların birleşimi (union) nihai gölge maskesi olur. "
             "'Yok' seçiliyse sadece parlaklık eşiği kullanılır."
    )
    shadow_text_prompt = st.sidebar.text_input(
        "Gölge Metin Prompt", value="shadow, dark shaded area",
        disabled=extra_shadow_method != "SAM3 (zero-shot metin prompt)",
        help="SAM3'e gölgeyi tarif eden metin promptu. Virgülle birden fazla ifade verebilirsin."
    )
    shadow_threshold = st.sidebar.slider(
        "Gölge Eşik Değeri (Shadow U-Net)", 0.1, 0.9, 0.5, 0.05,
        disabled=extra_shadow_method != "Shadow U-Net (eğitilmiş model)"
    )

    st.sidebar.header("SAM3 Prompt Kaynakları")
    add_box_prompt_search = st.sidebar.checkbox(
        "Box tabanlı SAM3 segmentasyonu (YOLO kutuları)", value=True,
        help="YOLO'dan gelen kutuları SAM3'e box prompt olarak verir. "
             "Kapatırsan SAM3 sadece aşağıdaki metin prompt taramasına güvenir "
             "(açıksa) ya da hiç çalışmaz."
    )
    add_text_prompt_search = st.sidebar.checkbox(
        "SAM3'ün metin promptuyla ek tarama yap", value=True,
        help="YOLO'nun box üretemediği binaları, SAM3'ün box'a ihtiyaç "
             "duymayan metin tabanlı (text-prompt) modu ile ayrıca arar. "
             "Box tabanlı akışı bozmaz, ek bir kaynak olarak eklenir."
    )
    text_prompt_value = st.sidebar.text_input(
        "Metin Prompt", value="rooftop, building",
        disabled=not add_text_prompt_search,
        help="Birden fazla kavramı virgülle ayırarak tek seferde arayabilirsin "
             "(örn. 'rooftop, building'). SAM3 bunları tek geçişte işler. "
             "Nadir (tepeden) uydu görüntülerinde genelde 'rooftop', "
             "'building'den daha iyi sonuç verir. Çok fazla eş anlamlı eklemek "
             "(3-4+ terim) aynı bina için birden fazla örtüşen maske üretip "
             "komşu binaların yanlışlıkla birleşmesine (blob'laşma) yol açabilir - "
             "1-2 terimle başlamak daha güvenli."
    )

    st.sidebar.header("SAM3 Dilimleme (Geniş Alan)")
    use_sliced_sam = st.sidebar.checkbox(
        "SAM3'ü dilimleyerek çalıştır (geniş alanlarda önerilir)", value=True,
        help="YOLO'daki gibi görüntüyü örtüşmeli dilimlere bölüp SAM3'ü her dilimde "
             "sadece o dilime düşen kutularla çalıştırır. Geniş alanlarda SAM3'ün "
             "geçersiz/boş maske üretmesini engellemeyi amaçlar."
    )
    sam_slice_size = st.sidebar.slider(
        "SAM3 Dilim Boyutu (px)", 512, 2048, 1024, 128,
        disabled=not use_sliced_sam,
        help="Her SAM3 çağrısına verilen dilimin piksel boyutu. Modelin genelde eğitildiği "
             "çözünürlüğe (~1024) yakın tutmak en tutarlı sonucu verir."
    )
    sam_overlap_ratio = st.sidebar.slider(
        "SAM3 Dilim Örtüşmesi", 0.0, 0.5, 0.4, 0.05,
        disabled=not use_sliced_sam,
        help="Dilimler arası örtüşme oranı. Sınırda kalan bir bina, örtüşme payı "
             "kendi piksel boyutundan büyükse en az bir dilimde TAM görünür ve "
             "iki parçaya bölünmez - dolayısıyla 'çift maske' oluşmaz. Küçük "
             "değerlerde bina iki dilimin sınırına denk gelip iki ayrı (yeterince "
             "örtüşmeyen, dolayısıyla birleştirilemeyen) maskeye bölünebilir."
    )

    st.sidebar.header("Fine-Tune Edilmiş SAM3 (Ek Kaynak)")
    use_finetuned_sam3 = st.sidebar.checkbox(
        "Fine-tune SAM3 modelini ek kaynak olarak kullan", value=False,
        help="Kendi bina veri setinizle fine-tune edilmiş SAM3 checkpoint'ini "
             "(Hungarian matching / instance segmentation) mevcut YOLO+SAM3 akışına "
             "EK bir kaynak olarak katar. Kapalıyken model hiç yüklenmez, "
             "VRAM/hız etkilenmez — istediğiniz zaman bu kutucuktan açıp kapatabilirsiniz."
    )
    finetuned_sam3_checkpoint_path = st.sidebar.text_input(
        "Fine-tune Checkpoint Yolu (.pth)",
        value=r"C:\1.Projeler\yolo-sam\sam3_checkpoints\best_sam3_building.pth",
        disabled=not use_finetuned_sam3,
        help="Google Drive'dan indirdiğiniz best_sam3_building.pth dosyasının yerel yolu."
    )
    finetuned_sam3_prompt = st.sidebar.text_input(
        "Fine-tune SAM3 Prompt", value="building",
        disabled=not use_finetuned_sam3,
        help="Eğitim sırasında prompt search ile seçilen prompt ile AYNI olmalı "
             "(checkpoint yanındaki prompt.txt dosyasına bakın)."
    )
    finetuned_sam3_confidence = st.sidebar.slider(
        "Fine-tune SAM3 Confidence Eşiği", 0.05, 0.9, 0.2, 0.05,
        disabled=not use_finetuned_sam3,
        help="Eğitim sırasındaki threshold sweep'te en iyi F1'i veren eşikle "
             "aynı tutmanız (genelde 0.2 civarı) önerilir."
    )
    if use_finetuned_sam3:
        if os.path.exists(finetuned_sam3_checkpoint_path):
            st.sidebar.success("✅ Fine-tune checkpoint bulundu.")
        else:
            st.sidebar.error(f"⚠️ Checkpoint bulunamadı: `{finetuned_sam3_checkpoint_path}`")

    st.sidebar.header("Görselleştirme & Geometri İyileştirme")
    mask_color = st.sidebar.color_picker("Maske Rengi (Özel)", "#1CF31C")
    apply_simplify = st.sidebar.checkbox("Kenarları düzleştir (Vektör Simplify)", value=True)
    simplify_level = st.sidebar.slider(
        label="Düzleştirme Toleransı (Metre)",
        min_value=0.1,
        max_value=2.0,
        value=0.3,
        step=0.1,
        help="Değeri artırdıkça pikselleşme ve tırtıklar kaybolur, düz kenarlar elde edilir. "
             "Bu tolerans artık doğrudan regularize() adımına aktarılır (tek adımlı geometri iyileştirme).",
        disabled=not apply_simplify,
    )


    settings = DetectionSettings(
        nms_threshold=nms_threshold,
        edge_buffer_pct=edge_buffer_pct,
        extra_shadow_method=extra_shadow_method,
        shadow_text_prompt=shadow_text_prompt,
        shadow_threshold=shadow_threshold,
        add_box_prompt_search=add_box_prompt_search,
        add_text_prompt_search=add_text_prompt_search,
        text_prompt_value=text_prompt_value,
        use_sliced_sam=use_sliced_sam,
        sam_slice_size=sam_slice_size,
        sam_overlap_ratio=sam_overlap_ratio,
        use_finetuned_sam3=use_finetuned_sam3,
        finetuned_sam3_checkpoint_path=finetuned_sam3_checkpoint_path,
        finetuned_sam3_prompt=finetuned_sam3_prompt,
        finetuned_sam3_confidence=finetuned_sam3_confidence,
        apply_simplify=apply_simplify,
        simplify_level=simplify_level,
    )

    return settings, map_source, mask_color
