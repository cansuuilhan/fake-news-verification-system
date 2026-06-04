import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.models.result_model import AnalysisResult
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.retrieval_service import RetrievalService
from app.services.verification_service import VerificationService
from app.utils.text_cleaner import clean_text


class AnalysisService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.claim_extraction_service = ClaimExtractionService()
        self.retrieval_service = RetrievalService()
        self.verification_service = VerificationService()
        self._train_model()

    def _train_model(self):
        data = pd.read_csv("data/fake_real_tr/turkish_fake_real.csv")

        # Dataset kolonları:
        # label      -> 0 / 1 etiketi
        # clean_data -> temizlenmiş haber metni
        data = data[["clean_data", "label"]].dropna()

        data["content"] = data["clean_data"].astype(str).str.lower()

        x = data["content"]
        y = data["label"]

        self.vectorizer = TfidfVectorizer()
        x_vectorized = self.vectorizer.fit_transform(x)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(x_vectorized, y)

        self.is_trained = True

    def analyze_news(self, text: str) -> AnalysisResult:
        cleaned_text = clean_text(text)

        if not cleaned_text:
            return AnalysisResult(
                label="Geçersiz Girdi",
                confidence=0.0,
                explanation="Metin boş olduğu için analiz yapılamadı.",
                claims=[],
                evidence={},
                verifications={}
            )

        if not self.is_trained:
            return AnalysisResult(
                label="Sistem Hatası",
                confidence=0.0,
                explanation="Model henüz hazır değil.",
                claims=[],
                evidence={},
                verifications={}
            )

        text_vectorized = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]

        extracted_claims = self.claim_extraction_service.extract_claims(text)

        evidence_map = {}
        verification_map = {}

        for claim in extracted_claims:
            evidences = self.retrieval_service.retrieve(claim, top_k=3)
            evidence_map[claim] = evidences

            verification_result = self.verification_service.verify_claim(
                claim,
                evidences
            )
            verification_map[claim] = verification_result

        if prediction == 0:
            confidence = probabilities[0] * 100
            return AnalysisResult(
                label="Sahte Haber olabilir",
                confidence=confidence,
                explanation="Türkçe veri seti ile eğitilen model, bu içeriği sahte habere daha yakın buldu.",
                claims=extracted_claims,
                evidence=evidence_map,
                verifications=verification_map
            )

        confidence = probabilities[1] * 100
        return AnalysisResult(
            label="Gerçek Haber olabilir",
            confidence=confidence,
            explanation="Türkçe veri seti ile eğitilen model, bu içeriği gerçek habere daha yakın buldu.",
            claims=extracted_claims,
            evidence=evidence_map,
            verifications=verification_map
        )