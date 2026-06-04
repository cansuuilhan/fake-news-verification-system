import re
from typing import List


class VerificationService:
    def __init__(self):
        self.stopwords = {
            "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de", "ki",
            "için", "gibi", "daha", "en", "çok", "az", "olarak", "olan",
            "oldu", "olduğu", "olduğunu", "ise", "ama", "fakat", "ancak",
            "göre", "kadar", "sonra", "önce", "her", "şey", "mi", "mı",
            "mu", "mü", "ya", "hem", "ise", "üzerine", "tarafından"
        }

    def verify_claim(self, claim: str, evidences: List[str]) -> dict:
        if not evidences:
            return {
                "verdict": "Yetersiz Kanıt",
                "score": 0.0,
                "explanation": "Bu iddia için yeterli kanıt bulunamadı.",
                "best_evidence": ""
            }

        claim_tokens = self._tokenize(claim)

        if not claim_tokens:
            return {
                "verdict": "Yetersiz Kanıt",
                "score": 0.0,
                "explanation": "İddia metni doğrulama için yeterli anlamlı kelime içermiyor.",
                "best_evidence": ""
            }

        best_score = 0.0
        best_evidence = ""

        for evidence in evidences:
            evidence_tokens = self._tokenize(evidence)

            if not evidence_tokens:
                continue

            score = self._calculate_similarity_score(
                claim_tokens,
                evidence_tokens
            )

            if score > best_score:
                best_score = score
                best_evidence = evidence

        percentage_score = round(best_score * 100, 2)

        if best_score >= 0.45:
            return {
                "verdict": "Destekleniyor",
                "score": min(percentage_score, 95),
                "explanation": "İddia ile bulunan kanıt arasında güçlü metinsel benzerlik tespit edildi.",
                "best_evidence": best_evidence[:300]
            }

        if best_score >= 0.20:
            return {
                "verdict": "Kısmen Destekleniyor.",
                "score": percentage_score,
                "explanation": "İddia ile kanıt arasında kısmi benzerlik bulundu ancak kesin doğrulama yapılamadı.",
                "best_evidence": best_evidence[:300]
            }

        return {
            "verdict": "Kanıt Bulunamadı.",
            "score": percentage_score,
            "explanation": "Bulunan kanıtlar iddiayı doğrulamak için yeterli değildir.",
            "best_evidence": best_evidence[:300]
        }

    def _tokenize(self, text: str) -> set:
        text = text.lower()
        text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
        words = text.split()

        meaningful_words = {
            word for word in words
            if len(word) > 2 and word not in self.stopwords
        }

        return meaningful_words

    def _calculate_similarity_score(self, claim_tokens: set, evidence_tokens: set) -> float:
        common_words = claim_tokens.intersection(evidence_tokens)

        if not common_words:
            return 0.0

        claim_coverage = len(common_words) / len(claim_tokens)
        evidence_coverage = len(common_words) / len(evidence_tokens)

        final_score = (claim_coverage * 0.75) + (evidence_coverage * 0.25)

        return final_score