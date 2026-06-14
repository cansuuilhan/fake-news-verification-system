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

        # Nokta, soru işareti, ünlem ve iki nokta sonrası bölme
        sentences = re.split(r'(?<=[.!?])\s+|(?<=:)\s+', text)

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _clean_sentences(self, sentences: List[str]) -> List[str]:
        cleaned = []

        noise_keywords = [
            "hikayen",
            "takip et",
            "senin için önerilenler",
            "beğen",
            "yorum",
            "paylaş",
            "reklam",
            "sponsorlu",
            "gaga haber",
            "gagahaber",
            "son dakika",
            "whatsapp",
            "instagram",
            "5g",
            "sn"
        ]

        for sentence in sentences:
            sentence = " ".join(sentence.split())
            sentence_lower = sentence.lower()

            if len(sentence) < 35:
                continue

            if any(noise in sentence_lower for noise in noise_keywords):
                # Eğer cümle tamamen sosyal medya arayüzü gibi görünüyorsa alma
                if len(sentence.split()) < 18:
                    continue

            if self._looks_like_garbage(sentence):
                continue

            cleaned.append(sentence)

        return cleaned

    def _looks_like_garbage(self, sentence: str) -> bool:
        words = sentence.split()

        if not words:
            return True

        if len(words) < 5:
            return True

        long_words = [
            word for word in words
            if len(word) > 22
        ]

        if len(long_words) >= 2:
            return True

        meaningful_words = [
            word for word in words
            if len(word) >= 3 and re.search(r"[aeıioöuüAEIİOÖUÜ]", word)
        ]

        meaningful_ratio = len(meaningful_words) / len(words)

        if meaningful_ratio < 0.55:
            return True

        return False

    def _score_sentence(self, sentence: str) -> int:
        score = 0
        sentence_lower = sentence.lower()

        claim_keywords = [
            "iddia",
            "ifade",
            "açıkladı",
            "belirtti",
            "söyledi",
            "duyurdu",
            "ortaya çıktı",
            "görevden",
            "tutuklama",
            "öldürdü",
            "öldüren",
            "vuruldu",
            "vurdu",
            "yakalandı",
            "soruşturma",
            "karar",
            "talimat",
            "para aldı",
            "hedef",
            "çete",
            "örgüt",
            "mahkeme",
            "başsavcı"
        ]

        for keyword in claim_keywords:
            if keyword in sentence_lower:
                score += 3

        # Sayı, para, tarih gibi somut bilgi içeriyorsa
        if re.search(r"\d+", sentence):
            score += 2

        if re.search(r"(tl|bin|milyon|yüzde|derece|kişi|ülke)", sentence_lower):
            score += 2

        # Büyük harfle başlayan özel isimler
        proper_nouns = re.findall(
            r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b",
            sentence
        )

        if len(proper_nouns) >= 2:
            score += 3
        elif len(proper_nouns) == 1:
            score += 1

        # Çok uzun cümleleri biraz cezalandır
        word_count = len(sentence.split())

        if 8 <= word_count <= 35:
            score += 2
        elif word_count > 45:
            score -= 2

        return score

    def _select_candidate_claims(
        self,
        sentences: List[str],
        max_claims: int
    ) -> List[str]:
        if not sentences:
            return []

        scored_sentences = []

        for sentence in sentences:
            score = self._score_sentence(sentence)
            scored_sentences.append((sentence, score))

        scored_sentences = sorted(
            scored_sentences,
            key=lambda item: item[1],
            reverse=True
        )

        selected_claims = []

        for sentence, score in scored_sentences:
            if score <= 0:
                continue

            if sentence not in selected_claims:
                selected_claims.append(sentence)

            if len(selected_claims) >= max_claims:
                break

        # Eğer hiç güçlü iddia bulunamazsa en anlamlı ilk cümleleri al
        if not selected_claims:
            selected_claims = sentences[:max_claims]

        return selected_claims