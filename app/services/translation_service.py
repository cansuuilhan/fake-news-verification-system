from deep_translator import GoogleTranslator


class TranslationService:
    def __init__(self):
        self.translator_to_en = GoogleTranslator(source="auto", target="en")
        self.translator_to_tr = GoogleTranslator(source="auto", target="tr")

    def translate_to_english(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        try:
            return self.translator_to_en.translate(text)
        except Exception:
            return text

    def translate_to_turkish(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        try:
            return self.translator_to_tr.translate(text)
        except Exception:
            return text