import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from app.models.result_model import AnalysisResult
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.retrieval_service import RetrievalService
from app.services.verification_service import VerificationService
from app.utils.text_cleaner import clean_text, is_low_quality_text


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
        data = data[["clean_data", "label"]].dropna()

        data["content"] = data["clean_data"].astype(str).str.lower()

        x = data["content"]
        y = data["label"]

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2)
        )

        x_vectorized = self.vectorizer.fit_transform(x)

        base_model = LinearSVC()
        self.model = CalibratedClassifierCV(base_model)
        self.model.fit(x_vectorized, y)

        self.is_trained = True

    def _generate_summary(self, text: str) -> str:
        cleaned = clean_text(text)

        sentences = [
            sentence.strip()
            for sentence in cleaned.replace("\n", " ").split(".")
            if len(sentence.strip()) > 35
        ]

        if not sentences:
            words = cleaned.split()
            if len(words) >= 20:
                return " ".join(words[:35]) + "..."
            return "Metin kısa olduğu için özet oluşturulamadı."

        clean_sentences = []

        for sentence in sentences:
            words = sentence.split()

            if len(words) < 5:
                continue

            meaningful_words = [
                word for word in words
                if len(word) >= 3
            ]

            if len(meaningful_words) / len(words) >= 0.60:
                clean_sentences.append(sentence)

        if not clean_sentences:
            words = cleaned.split()
            if len(words) >= 20:
                return " ".join(words[:35]) + "..."
            return "Metin kalitesi düşük olduğu için sağlıklı özet oluşturulamadı."

        summary_sentences = clean_sentences[:2]
        return ". ".join(summary_sentences) + "."

    def _calculate_average_verification_score(self, verification_map: dict) -> float:
        verification_scores = [
            result.get("score", 0)
            for result in verification_map.values()
        ]

        if not verification_scores:
            return 0.0

        return sum(verification_scores) / len(verification_scores)

    def _calculate_final_decision(
        self,
        label: str,
        confidence: float,
        verification_map: dict
    ):
        avg_verification_score = self._calculate_average_verification_score(
            verification_map
        )

        final_score = (
            confidence * 0.30
            + avg_verification_score * 0.70
        )

        final_score = round(final_score, 2)

        if "Sahte" in label:
            return (
                "🚨 Şüpheli / Sahte Haber Olabilir",
                min(final_score, 40.0)
            )

        if "Gerçek" in label:
            if avg_verification_score >= 70:
                return (
                    "Büyük Ölçüde Güvenilir",
                    final_score
                )

            if avg_verification_score >= 55:
                return (
                    "Kısmen Güvenilir: Model metni gerçek habere yakın buldu; "
                    "ancak kanıt desteği tam güçlü değildir. Ek kaynak kontrolü önerilir.",
                    min(final_score, 65.0)
                )

            return (
                "⚠️ Şüpheli / Teyit Edilemedi: Model metni gerçek haber diline "
                "benzetti; ancak iddiayı destekleyen yeterli ve güçlü kanıt bulunamadı. "
                "Bu nedenle haber güvenilir kabul edilmemelidir.",
                min(final_score, 45.0)
            )

        return "Nihai karar üretilemedi.", final_score

    def _should_reject_as_low_quality(self, cleaned_text: str) -> bool:
        word_count = len(cleaned_text.split())

        # Uzun haberleri düşük kaliteli diye kesme.
        # Dataset clean_data metinleri noktasız ve köklenmiş olabilir.
        if word_count >= 40:
            return False

        return is_low_quality_text(cleaned_text)

    def analyze_news(self, text: str) -> AnalysisResult:
        cleaned_text = clean_text(text)

        if not cleaned_text:
            return AnalysisResult(
                label="Geçersiz Girdi",
                confidence=0.0,
                explanation="Metin boş olduğu için analiz yapılamadı.",
                summary="",
                final_decision="Geçersiz girdi.",
                final_score=0.0,
                claims=[],
                evidence={},
                verifications={}
            )

        if self._should_reject_as_low_quality(cleaned_text):
            return AnalysisResult(
                label="Düşük Kaliteli Metin",
                confidence=0.0,
                explanation=(
                    "Metin yeterince anlamlı değil. Daha net bir görsel yükleyin "
                    "veya metni manuel düzenleyin."
                ),
                summary="Metin kalitesi düşük olduğu için özet oluşturulmadı.",
                final_decision=(
                    "Analiz yapılamadı. Girdi metni OCR hataları veya anlamsız "
                    "kelimeler içeriyor."
                ),
                final_score=0.0,
                claims=[],
                evidence={},
                verifications={}
            )

        if not self.is_trained:
            return AnalysisResult(
                label="Sistem Hatası",
                confidence=0.0,
                explanation="Model henüz hazır değil.",
                summary="",
                final_decision="Sistem modeli hazır değil.",
                final_score=0.0,
                claims=[],
                evidence={},
                verifications={}
            )

        text_vectorized = self.vectorizer.transform([cleaned_text])

        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]

        extracted_claims = self.claim_extraction_service.extract_claims(cleaned_text)

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

        summary = self._generate_summary(cleaned_text)

        if prediction == 0:
            label = "Sahte Haber olabilir"
            confidence = probabilities[0] * 100
            explanation = (
                "Linear SVM modeli, bu içeriği sahte haber sınıfına daha yakın buldu."
            )
        else:
            label = "Gerçek Haber olabilir"
            confidence = probabilities[1] * 100
            explanation = (
                "Linear SVM modeli, bu içeriği gerçek haber sınıfına daha yakın buldu."
            )

        final_decision, final_score = self._calculate_final_decision(
            label,
            confidence,
            verification_map
        )

        if "Şüpheli" in final_decision or "Teyit Edilemedi" in final_decision:
            label = "Sahte Haber olabilir"
            confidence = max(75.0, 100 - final_score)
            explanation = (
                "Model metni haber diline benzetmiş olsa da doğrulama katmanı "
                "iddiayı destekleyen yeterli ve güçlü kanıt bulamadı."
            )

        return AnalysisResult(
            label=label,
            confidence=confidence,
            explanation=explanation,
            summary=summary,
            final_decision=final_decision,
            final_score=final_score,
            claims=extracted_claims,
            evidence=evidence_map,
            verifications=verification_map
        )