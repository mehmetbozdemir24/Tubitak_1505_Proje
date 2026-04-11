import re
from typing import Tuple


TR_CHARS = "çğıöşüÇĞİÖŞÜ"

TR_STOPWORDS = {
    "ve", "veya", "ile", "ama", "fakat", "için", "icin", "gibi", "daha", "çok", "cok", "az",
    "bir", "bu", "şu", "su", "o", "biz", "siz", "onlar", "ben", "sen", "mi", "mı", "mu", "mü",
    "mi", "nedir", "neden", "nasıl", "nasil", "hangi", "kadar", "de", "da", "ki", "olarak",
    "sistem", "doküman", "dokuman", "cevap", "soru", "bilgi", "güncel", "guncel", "dosya",
    "kim", "kimler", "kimden", "kimlerden", "nerede", "nereden", "komisyon", "komisyonu",
    "olusur", "oluşur", "rapor", "yonetmelik", "talimat", "kurul", "uygulama"
}

EN_STOPWORDS = {
    "the", "and", "or", "but", "for", "with", "without", "about", "from", "into", "on", "in",
    "a", "an", "is", "are", "was", "were", "be", "to", "of", "that", "this", "these", "those",
    "what", "why", "how", "which", "when", "where", "who", "whom", "can", "could", "should",
    "document", "answer", "question", "information", "system", "file", "please", "summary",
    "budget", "committee", "policy", "report", "guideline", "procedure", "project", "cost"
}

TR_SUFFIXES = (
    "lar", "ler", "dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür",
    "lik", "lık", "luk", "lük", "cı", "ci", "cu", "cü", "yor", "acak", "ecek",
    "miş", "mış", "muş", "müş", "dan", "den", "nin", "nın", "nun", "nün",
    "mak", "mek", "siniz", "umuz", "imiz", "lari", "leri"
)


def _tokenize(text: str):
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+", text.lower())


def _language_scores(text: str) -> Tuple[float, float]:
    if not text:
        return 0.0, 0.0

    sample = text.strip()[:8000]
    if not sample:
        return 0.0, 0.0

    tokens = _tokenize(sample)
    if not tokens:
        return 0.0, 0.0

    tr_score = 0.0
    en_score = 0.0

    tr_char_count = sum(sample.count(ch) for ch in TR_CHARS)
    if tr_char_count:
        tr_score += 2.5 + min(tr_char_count * 0.35, 5.0)

    if re.search(r"\b(mi|mı|mu|mü|nasil|neden|hangi|kim|kimler|kimlerden)\b", sample.lower()):
        tr_score += 1.2

    ascii_token_count = 0
    for token in tokens[:1200]:
        if token in TR_STOPWORDS:
            tr_score += 1.4
        if token in EN_STOPWORDS:
            en_score += 1.4

        if len(token) >= 4 and token.endswith(TR_SUFFIXES):
            tr_score += 0.6

        if len(token) >= 4 and (
            token.endswith("ing")
            or token.endswith("ed")
            or token.endswith("tion")
            or token.endswith("ment")
            or token.endswith("ity")
        ):
            en_score += 0.3

        if token.isascii():
            ascii_token_count += 1

    if ascii_token_count and tr_char_count == 0:
        en_score += min(ascii_token_count / 80.0, 2.0)

    return tr_score, en_score


def detect_language(text: str) -> str:
    tr_score, en_score = _language_scores(text)
    if tr_score == 0.0 and en_score == 0.0:
        return "unknown"

    diff = tr_score - en_score
    if diff >= 0.9:
        return "tr"
    if diff <= -0.9:
        return "en"
    return "unknown"


def choose_answer_language(question: str, context_text: str, default_language: str = "tr"):
    question_lang = detect_language(question)
    context_lang = detect_language(context_text)

    # Kullanıcı sorusunun dili birincil sinyaldir; belirsizse bağlam diline düşeriz.
    if question_lang in {"tr", "en"}:
        return question_lang, question_lang, context_lang, "question"
    if context_lang in {"tr", "en"}:
        return context_lang, question_lang, context_lang, "context"
    return default_language, question_lang, context_lang, "default"


def get_no_answer_message(language: str) -> str:
    if language == "en":
        return "I could not find relevant information in the documents you are authorized to access."
    return "Yetkiniz dahilindeki dokumanlarda bu konuyla ilgili bilgi bulamadim."


def build_language_policy_prompt(language: str) -> str:
    if language == "en":
        return (
            "Language policy:\n"
            "- You must answer entirely in English.\n"
            "- If the evidence is Turkish, translate it into English while preserving meaning.\n"
            "- If the answer is not in the context, say exactly: I don't know."
        )

    return (
        "Dil politikasi:\n"
        "- Yaniti tamamen Turkce ver.\n"
        "- Kanit metni Ingilizce ise anlami bozmadan Turkceye cevirerek aktar.\n"
        "- Cevap baglamda yoksa tam olarak sunu soyle: Bilgim yok."
    )


def get_language_label(language: str) -> str:
    if language == "en":
        return "English"
    if language == "tr":
        return "Turkce"
    return "Bilinmiyor"