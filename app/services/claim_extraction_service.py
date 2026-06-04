import re
from typing import List


class ClaimExtractionService:
    def extract_claims(self, text: str, max_claims: int = 3) -> List[str]:
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        cleaned_sentences = self._clean_sentences(sentences)
        claims = self._select_candidate_claims(cleaned_sentences, max_claims)

        return claims

    def _split_into_sentences(self, text: str) -> List[str]:
        text = text.replace("\n", " ").strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _clean_sentences(self, sentences: List[str]) -> List[str]:
        cleaned = []

        for sentence in sentences:
            sentence = " ".join(sentence.split())

            if len(sentence) < 30:
                continue

            cleaned.append(sentence)

        return cleaned

    def _select_candidate_claims(self, sentences: List[str], max_claims: int) -> List[str]:
        if not sentences:
            return []

        # Şimdilik en uzun ve anlamlı ilk cümlelerden seçiyoruz
        sorted_sentences = sorted(sentences, key=len, reverse=True)
        return sorted_sentences[:max_claims]