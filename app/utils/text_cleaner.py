import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9.,!?%:;'\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_low_quality_text(text: str) -> bool:
    cleaned = clean_text(text)

    if len(cleaned) < 80:
        return True

    words = cleaned.split()

    if len(words) < 12:
        return True

    meaningful_words = [
        word for word in words
        if len(word) >= 3 and re.search(r"[aeıioöuüAEIİOÖUÜ]", word)
    ]

    meaningful_ratio = len(meaningful_words) / len(words)

    if meaningful_ratio < 0.55:
        return True

    long_garbage_words = [
        word for word in words
        if len(word) > 18
    ]

    if len(long_garbage_words) >= 3:
        return True

    return False