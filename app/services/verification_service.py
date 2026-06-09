import re
from typing import List
from sentence_transformers import SentenceTransformer, util


class VerificationService:
    def __init__(self):
        # Kelime saymak yerine anlamsal vektör çıkaran yapay zekâ modelini tanımlıyoruz
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def verify_claim(self, claim: str, evidences: List[str]) -> dict:
        if not evidences:
            return {
                "verdict": "Yetersiz Kanıt",
                "score": 0.0,
                "explanation": "Bu iddia için veri tabanında hiçbir kaynak veya kanıt bulunamadı.",
                "best_evidence": ""
            }

        if not claim or not claim.strip():
            return {
                "verdict": "Yetersiz Kanıt",
                "score": 0.0,
                "explanation": "İddia metni doğrulama için yeterli anlamlı kelime içermiyor.",
                "best_evidence": ""
            }

        # İddianın anlamsal vektörünü (embedding) hesaplıyoruz
        claim_embedding = self.embedding_model.encode(claim, convert_to_tensor=True)

        best_score = 0.0
        best_evidence = ""

        # Kanıtları vektörel kosinüs benzerliğine göre tarıyoruz
        for evidence in evidences:
            if not evidence or not evidence.strip():
                continue

            evidence_embedding = self.embedding_model.encode(evidence, convert_to_tensor=True)
            similarity = util.cos_sim(claim_embedding, evidence_embedding).item()

            if similarity > best_score:
                best_score = similarity
                best_evidence = evidence

        percentage_score = round(best_score * 100, 2)

        # ÇAPRAZ KONTROL FİLTRESİ: ÖZEL İSİM KONTROLÜ
        # İddiadaki 4 harften uzun, büyük harfle başlayan kelimeleri yakalar (Örn: Can, Polat, Daltonlar)
        proper_nouns = re.findall(r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{3,}\b', claim)
        
        # Eğer iddiada özel isim varsa ve bu isim bulunan en güçlü kanıtta HİÇ geçmiyorsa skoru cezalandır
        if proper_nouns and best_evidence:
            matches = [noun for noun in proper_nouns if noun.lower() in best_evidence.lower()]
            if len(matches) == 0:
                # İsim eşleşmesi başarısız olduğu için skoru %40'ına düşürerek alakayı koparırız
                best_score = best_score * 0.4 
                percentage_score = round(best_score * 100, 2)

        # YENİ VE SERTLEŞTİRİLMİŞ KARAR EŞİKLERİ
        if best_score >= 0.68:
            return {
                "verdict": "Destekleniyor",
                "score": min(percentage_score, 98.5),
                "explanation": "Yapay zekâ modeli, iddia ile bulunan kaynak kaynak arasında çok güçlü bir anlamsal bağ tespit etti.",
                "best_evidence": best_evidence[:300]
            }

        if best_score >= 0.50:
            return {
                "verdict": "Kısmen Destekleniyor",
                "score": percentage_score,
                "explanation": "İddia ile kanıt arasında kısmi anlamsal benzerlikler bulundu ancak kesin yargı için yeterli değildir.",
                "best_evidence": best_evidence[:300]
            }

        return {
            "verdict": "Kanıt Bulunamadı / Alakasız",
            "score": percentage_score,
            "explanation": "Arşivde bulunan belgeler bu iddiayı doğrulamak için yetersizdir veya içerik tamamen alakasızdır.",
            "best_evidence": best_evidence[:300]
        }