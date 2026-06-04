import streamlit as st
import PyPDF2
from app.services.analysis_service import AnalysisService


st.set_page_config(page_title="Güvenilir Haber Doğrulama", layout="wide")


NEWS_THEME = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
        font-family: 'Inter', sans-serif;
    }

    .header-section {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        box-shadow: 0 35px 60px -15px rgba(30, 64, 175, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2.5rem;
    }

    .input-card, .result-card {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.1);
        margin-bottom: 1.5rem;
    }

    .status-badge {
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        color: white;
    }

    .real-badge {
        background: linear-gradient(135deg, #10b981, #059669);
    }

    .fake-badge {
        background: linear-gradient(135deg, #ef4444, #dc2626);
    }

    .metric-container {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .claim-item {
        background: rgba(248, 250, 252, 0.9);
        border-left: 4px solid #3b82f6;
        padding: 1.3rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.1);
    }

    .evidence-item {
        background: rgba(248, 250, 252, 0.8);
        border-left: 4px solid #10b981;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }

    .verification-item {
        background: rgba(248, 250, 252, 0.9);
        border-left: 4px solid #6366f1;
        padding: 1.3rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }

    .section-title {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }

    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid rgba(59, 130, 246, 0.2) !important;
        padding: 1.2rem !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 14px;
        height: 3rem;
        font-weight: 700;
        width: 100%;
    }

    div[data-testid="stButton"] > button:hover {
        color: white;
        border: none;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
</style>
"""


@st.cache_resource
def get_analysis_service():
    return AnalysisService()


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

    return ""


def run_app():
    st.markdown(NEWS_THEME, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-section">
        <h1 style="color:white; font-size:3rem; font-weight:800; margin:0;">
            📰 Güvenilir Haber Doğrulama
        </h1>
        <p style="font-size:1.2rem; color:rgba(255,255,255,0.92); margin-top:1rem;">
            Yapay zekâ destekli haber analizi ile metinleri değerlendirin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    service = get_analysis_service()

    st.markdown("""
    <div class="input-card">
        <h3 class="section-title">Girdi Seçimi</h3>
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
            height=300,
            placeholder="Analiz etmek istediğiniz haber metnini buraya yapıştırın..."
        )
    else:
        uploaded_file = st.file_uploader(
            "TXT veya PDF dosyası yükleyin",
            type=["txt", "pdf"]
        )

        if uploaded_file is not None:
            news_text = read_uploaded_file(uploaded_file)

            st.text_area(
                "Yüklenen Dosyadan Çıkarılan Metin",
                value=news_text,
                height=300
            )

    if st.button("🚀 ANALİZ ET", key="analyze"):
        if not news_text.strip():
            st.error("Lütfen haber metni girin veya dosya yükleyin.")
            return

        with st.spinner("Yapay zekâ analiz ediyor..."):
            result = service.analyze_news(news_text)

        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

        if result.label == "Geçersiz Girdi":
            st.markdown(f"""
            <div class="result-card" style="border-left:6px solid #f59e0b; text-align:center;">
                <div style="font-size:4rem;">⚠️</div>
                <h3 style="color:#f59e0b;">{result.label}</h3>
                <p style="color:#64748b;">{result.explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            return

        if result.label == "Sistem Hatası":
            st.markdown(f"""
            <div class="result-card" style="border-left:6px solid #ef4444; text-align:center;">
                <div style="font-size:4rem;">💥</div>
                <h3 style="color:#ef4444;">{result.label}</h3>
                <p style="color:#64748b;">{result.explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            badge_class = "real-badge" if "Sahte" not in result.label else "fake-badge"
            icon = "✅" if "Sahte" not in result.label else "🚨"

            st.markdown(f"""
            <div class="result-card" style="text-align:center;">
                <div style="font-size:4rem; margin-bottom:1rem;">{icon}</div>
                <div class="status-badge {badge_class}">{result.label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-container">
                    <h3 style="color:#1e40af; margin:0; font-size:2.5rem;">
                        %{result.confidence:.1f}
                    </h3>
                    <p style="color:#3b82f6; font-weight:700; margin:0;">
                        GÜVEN ORANI
                    </p>
                </div>
                <div style="margin-top:1.5rem; padding:1.2rem; background:rgba(59,130,246,0.06); border-radius:12px; border-left:4px solid #3b82f6;">
                    <strong>Açıklama:</strong><br>
                    <span style="color:#475569;">{result.explanation}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if result.claims:
            st.markdown("""
            <div class="result-card">
                <h3 class="section-title">🎯 Tespit Edilen Ana İddialar</h3>
            </div>
            """, unsafe_allow_html=True)

            for index, claim in enumerate(result.claims, start=1):
                st.markdown(
                    f'<div class="claim-item"><strong>{index}.</strong> {claim}</div>',
                    unsafe_allow_html=True
                )

        if result.evidence:
            st.markdown("""
            <div class="result-card">
                <h3 class="section-title">📋 Bulunan Kanıtlar</h3>
            </div>
            """, unsafe_allow_html=True)

            for claim, evidences in result.evidence.items():
                st.markdown(
                    f'<div style="margin-bottom:1rem;"><strong style="color:#1e40af;">İddia:</strong> {claim}</div>',
                    unsafe_allow_html=True
                )

                if evidences:
                    for index, evidence in enumerate(evidences, start=1):
                        st.markdown(
                            f'<div class="evidence-item"><strong>{index}.</strong> {evidence[:300]}...</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div class="evidence-item" style="border-left-color:#f87171;">Bu iddia için kanıt bulunamadı.</div>',
                        unsafe_allow_html=True
                    )

        if result.verifications:
            st.markdown("""
            <div class="result-card">
                <h3 class="section-title">✅ Doğrulama Sonuçları</h3>
            </div>
            """, unsafe_allow_html=True)

            for claim, verification in result.verifications.items():
                verdict = verification.get("verdict", "Bilinmiyor")
                score = verification.get("score", 0)
                explanation = verification.get("explanation", "")
                best_evidence = verification.get("best_evidence", "")

                st.markdown(f"""
                <div class="verification-item">
                    <strong style="color:#1e40af;">İddia:</strong><br>
                    {claim}<br><br>
                    <strong>Karar:</strong> {verdict}<br>
                    <strong>Skor:</strong> %{score:.1f}<br>
                    <strong>Açıklama:</strong> {explanation}
                </div>
                """, unsafe_allow_html=True)

                if best_evidence:
                    st.markdown(
                        f'<div class="evidence-item"><strong>🔍 En Güçlü Kanıt:</strong><br>{best_evidence[:250]}...</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:#94a3b8;">
        <h4>⚡ Güvenilir Haber Doğrulama Sistemi</h4>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    run_app()