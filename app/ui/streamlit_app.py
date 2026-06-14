import streamlit as st
import PyPDF2
import easyocr
import numpy as np
from PIL import Image

from app.services.analysis_service import AnalysisService

st.set_page_config(page_title="Güvenilir Haber Doğrulama", layout="wide")


NEWS_THEME = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f7f3ea 0%, #f4efe6 45%, #eef2f3 100%);
        font-family: 'Inter', sans-serif;
        color: #1f2933;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }

    .header-section {
        background: linear-gradient(135deg, #233142 0%, #455d7a 100%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        box-shadow: 0 20px 45px -25px rgba(35, 49, 66, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 1.8rem;
    }

    .header-title {
        color: #ffffff;
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.03em;
    }

    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.86);
        margin-top: 0.7rem;
        margin-bottom: 0;
    }

    .input-card, .result-card {
        background: rgba(255, 252, 246, 0.96);
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 14px 35px -24px rgba(31, 41, 51, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.22);
        margin-bottom: 1.1rem;
    }

    .status-badge {
        padding: 0.75rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        color: white;
    }

    .real-badge {
        background: linear-gradient(135deg, #5f8d4e, #3f6f44);
    }

    .fake-badge {
        background: linear-gradient(135deg, #b85c5c, #9b4444);
    }

    .metric-container {
        background: linear-gradient(135deg, #eef3f2 0%, #e0ebe8 100%);
        border-radius: 15px;
        padding: 1.4rem;
        text-align: center;
        border: 1px solid rgba(69, 93, 122, 0.16);
    }

    .claim-item {
        background: rgba(250, 247, 240, 0.95);
        border-left: 4px solid #455d7a;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(69, 93, 122, 0.12);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .final-decision-card {
        background: linear-gradient(135deg, #fff7e6 0%, #f6ead5 100%);
        border-left: 5px solid #c08b5c;
        border-radius: 15px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 14px 35px -24px rgba(120, 83, 45, 0.45);
    }

    .summary-card {
        background: rgba(238, 246, 243, 0.98);
        border-left: 5px solid #6b9080;
        border-radius: 15px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.1rem;
    }

    .section-title {
        color: #233142;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.18rem;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.85rem;
    }

    .stTextArea textarea {
        border-radius: 14px !important;
        border: 1.5px solid rgba(69, 93, 122, 0.22) !important;
        padding: 1rem !important;
        background: rgba(255, 255, 255, 0.96) !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #455d7a 0%, #233142 100%);
        color: white;
        border: none;
        border-radius: 12px;
        height: 2.8rem;
        font-weight: 700;
        font-size: 0.95rem;
        width: 100%;
    }

    div[data-testid="stButton"] > button:hover {
        color: white;
        border: none;
        background: linear-gradient(135deg, #364b63 0%, #1f2933 100%);
    }

    div[data-testid="stRadio"] label {
        font-size: 0.92rem !important;
    }

    .footer {
        text-align:center;
        padding:2.2rem 0;
        color:#8b8b8b;
        font-size:0.9rem;
    }
</style>
"""


@st.cache_resource
def get_analysis_service():
    return AnalysisService()


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    return easyocr.Reader(["tr"], gpu=False)


def read_image_file(uploaded_file) -> str:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    reader = get_ocr_reader()

    ocr_results = reader.readtext(
        image_array,
        detail=0,
        paragraph=True
    )

    return "\n".join(ocr_results).strip()


def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")

    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        return read_image_file(uploaded_file)

    return ""


def show_verdict(verdict: str):
    if verdict == "Destekleniyor":
        st.markdown(
            """
            <div style="
                color: #233142;
                font-weight:700;
                font-size:1rem;
                margin:0.5rem 0;
            ">
                🟢 Destekleniyor
            </div>
            """,
            unsafe_allow_html=True
        )

    elif verdict == "Kısmen Destekleniyor":
        st.markdown(
            """
            <div style="
                color: #233142;
                font-weight:700;
                font-size:1rem;
                margin:0.5rem 0;
            ">
                🟡 Kısmen Destekleniyor
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="
                color: #233142;
                font-weight:700;
                font-size:1rem;
                margin:0.5rem 0;
            ">
                🔴 Kanıt Bulunamadı
            </div>
            """,
            unsafe_allow_html=True
        )


def run_app():
    st.markdown(NEWS_THEME, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-section">
        <h1 class="header-title">📰 Güvenilir Haber Doğrulama</h1>
        <p class="header-subtitle">
            Yapay zekâ destekli haber analizi ile metin, dosya ve görselleri değerlendirin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    service = get_analysis_service()

    st.markdown("""
    <div class="input-card">
        <h3 class="section-title">Girdi Seçimi</h3>
        <p class="small-muted">Analiz etmek istediğiniz haber metnini doğrudan yazabilir veya dosya yükleyebilirsiniz.</p>
    </div>
    """, unsafe_allow_html=True)

    input_type = st.radio(
        "Analiz türünü seçin:",
        ["Metin Gir", "Dosya Yükle"],
        horizontal=True
    )

    news_text = ""

    if input_type == "Metin Gir":
        news_text = st.text_area(
            "Haber Metni",
            height=260,
            placeholder="Analiz etmek istediğiniz haber metnini buraya yapıştırın..."
        )

    else:
        uploaded_file = st.file_uploader(
            "TXT, PDF veya görsel dosyası yükleyin",
            type=["txt", "pdf", "png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                col_image, col_ocr = st.columns([1, 1])

                with col_image:
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

                with col_ocr:
                    with st.spinner("Görselden Türkçe metin çıkarılıyor..."):
                        news_text = read_uploaded_file(uploaded_file)

                    st.text_area(
                        "Görselden Çıkarılan Metin",
                        value=news_text,
                        height=360
                    )

                    if not news_text.strip():
                        st.warning(
                            "Görselden metin çıkarılamadı. Görsel bulanık, düşük çözünürlüklü "
                            "veya metin içermiyor olabilir."
                        )

            else:
                news_text = read_uploaded_file(uploaded_file)

                st.text_area(
                    "Yüklenen Dosyadan Çıkarılan Metin",
                    value=news_text,
                    height=280
                )

                if not news_text.strip():
                    st.warning(
                        "Dosyadan metin çıkarılamadı. PDF taranmış olabilir veya dosya boş olabilir."
                    )

    if st.button("Analiz Et", key="analyze"):
        if not news_text.strip():
            st.error("Lütfen haber metni girin veya dosya yükleyin.")
            return

        with st.spinner("Yapay zekâ analiz ediyor..."):
            result = service.analyze_news(news_text)

        st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

        if result.label == "Geçersiz Girdi":
            st.warning(result.explanation)
            return

        if result.label == "Sistem Hatası":
            st.error(result.explanation)
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            badge_class = "real-badge" if "Sahte" not in result.label else "fake-badge"
            icon = "✅" if "Sahte" not in result.label else "🚨"

            st.markdown(f"""
            <div class="result-card" style="text-align:center;">
                <div style="font-size:2.8rem; margin-bottom:0.8rem;">{icon}</div>
                <div class="status-badge {badge_class}">{result.label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-container">
                    <h3 style="color:#233142; margin:0; font-size:2rem;">
                        %{result.confidence:.1f}
                    </h3>
                    <p style="color:#455d7a; font-weight:700; margin:0; font-size:0.85rem;">
                        SİSTEM DEĞERLENDİRME SKORU
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="final-decision-card">
            <h3 style="margin-top:0; color:#8a5a31; font-size:1.1rem;">🧭 Genel Nihai Karar</h3>
            <h2 style="color:#5f3d22; margin-bottom:0.5rem; font-size:1.25rem;">{result.final_decision}</h2>
            <p style="font-weight:700; color:#8a5a31; font-size:0.92rem; margin-bottom:0;">
                Genel Güven Skoru: %{result.final_score:.1f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if result.summary:
            st.markdown(f"""
            <div class="summary-card">
                <h3 style="margin-top:0; color:#41675a; font-size:1.1rem;">📝 Kısa Haber Özeti</h3>
                <p style="color:#334155; font-size:0.92rem; line-height:1.6; margin-bottom:0;">
                    {result.summary}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if result.claims:
            st.markdown("""
            <div class="result-card">
                <h3 class="section-title">🎯 Tespit Edilen Ana İddialar</h3>
                <p class="small-muted">
                    Sistem; kişi, kurum, olay, sayı, tarih veya net iddia içeren cümleleri öncelikli olarak seçer.
                </p>
            </div>
            """, unsafe_allow_html=True)

            for index, claim in enumerate(result.claims, start=1):
                st.markdown(
                    f'<div class="claim-item"><strong>{index}.</strong> {claim}</div>',
                    unsafe_allow_html=True
                )

        st.info(
            "Bu sistem bir karar destek aracıdır. Sonuçlar kesin hüküm niteliği taşımaz; "
            "özellikle kanıt bulunamayan veya düşük skorlu haberlerde ek kaynak kontrolü önerilir."
        )

        if "Sahte" in result.final_decision or "Şüpheli" in result.final_decision or "Teyit" in result.final_decision:
            st.warning("Sistem bu haber için şüpheli / teyit edilemedi sinyali üretmiştir.")
        elif "Güvenilir" in result.final_decision:
            st.success("Sistem bu haberin mevcut model ve arşiv kanıtlarıyla büyük ölçüde güvenilir göründüğünü belirtmektedir.")
        else:
            st.info("Sistem bu haber için kesin olmayan bir değerlendirme üretmiştir.")

        with st.expander("Analiz Detayları ve Kanıt Durumu", expanded=False):
            if result.verifications:
                for claim, verification in result.verifications.items():
                    verdict = verification.get("verdict", "Bilinmiyor")
                    score = verification.get("score", 0)
                    confidence_level = verification.get("confidence_level", "Bilinmiyor")
                    explanation = verification.get("explanation", "")
                    best_evidence = verification.get("best_evidence", "")

                    st.markdown("**İddia:**")
                    st.write(claim)

                    st.markdown(f"**Kanıt Durumu:** {verdict}")
                    st.markdown(f"**Güven Düzeyi:** {confidence_level} (%{score:.1f})")

                    if explanation:
                        st.caption(explanation)

                    if score >= 60 and best_evidence:
                        st.markdown("**İlgili arşiv metni:**")
                        st.info(best_evidence)
                    else:
                        st.info(
                            "Bu iddia için mevcut arşivde yeterince güçlü kanıt bulunamadı. "
                            "Bu durum haberin kesin olarak yanlış olduğu anlamına gelmez."
                        )

                    st.divider()
            else:
                st.info("Metinden doğrulanabilir açık bir iddia çıkarılamadı.")


if __name__ == "__main__":
    run_app()