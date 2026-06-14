import re
from typing import List
from sentence_transformers import SentenceTransformer, util


class VerificationService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def _get_confidence_level(self, score: float) -> str:
        if score >= 70:
            return "Yüksek"
        if score >= 40:
            return "Orta"
        return "Düşük"

    def _keyword_overlap_score(self, claim: str, evidence: str) -> float:
        claim_words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}", claim.lower()))
        evidence_words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}", evidence.lower()))

        if not claim_words:
            return 0.0

        important_words = {
            word for word in claim_words
            if word not in ["ancak", "sonra", "bunun", "olarak", "için", "olan", "olduğunu"]
        }

        if not important_words:
            return 0.0

        matched = important_words.intersection(evidence_words)
        return len(matched) / len(important_words)

    def verify_claim(self, claim: str, evidences: List[str]) -> dict:
        if not evidences:
            return {
                "verdict": "Yetersiz Kanıt",
                "score": 0.0,
                "confidence_level": "Düşük",
                "explanation": "Haber arşivinde bu iddiayı destekleyen yeterli bilgi bulunamadı. Bu sonuç haberin kesin olarak yanlış olduğu anlamına gelmez.",
                "best_evidence": ""
            }

        claim_embedding = self.embedding_model.encode(claim, convert_to_tensor=True)

        best_score = 0.0
        best_evidence = ""

        for evidence in evidences:
            evidence_embedding = self.embedding_model.encode(evidence, convert_to_tensor=True)
            semantic_score = util.cos_sim(claim_embedding, evidence_embedding).item()
            keyword_score = self._keyword_overlap_score(claim, evidence)

            final_score = (semantic_score * 0.70) + (keyword_score * 0.30)

            if final_score > best_score:
                best_score = final_score
                best_evidence = evidence

        percentage_score = round(best_score * 100, 2)
        confidence_level = self._get_confidence_level(percentage_score)

        if best_score >= 0.60:
            return {
                "verdict": "Destekleniyor",
                "score": percentage_score,
                "confidence_level": confidence_level,
                "explanation": "Arşivde bulunan en yakın haber metni, iddiadaki ana kişi/olay bilgileriyle büyük ölçüde örtüşüyor.",
                "best_evidence": best_evidence[:500]
            }

        if best_score >= 0.42:
            return {
                "verdict": "Kısmen Destekleniyor",
                "score": percentage_score,
                "confidence_level": confidence_level,
                "explanation": "Arşivde bu iddiaya benzeyen bazı bilgiler bulundu; ancak bulunan metin iddiayı tamamen doğrulayacak kadar güçlü değil.",
                "best_evidence": best_evidence[:500]
            }

        return {
            "verdict": "Kanıt Bulunamadı",
            "score": percentage_score,
            "confidence_level": confidence_level,
            "explanation": "Arşivde bu iddiayı doğrudan destekleyen yeterli bilgi bulunamadı. Bu haberin yanlış olduğu anlamına gelmez; yalnızca sistem mevcut arşivle doğrulayamamıştır.",
            "best_evidence": best_evidence[:500]
        }