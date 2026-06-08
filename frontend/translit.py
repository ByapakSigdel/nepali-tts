"""
Offline romanized-Nepali -> Devanagari for the TTS frontend. NO torch dependency
(indic-transliteration is pure-Python: regex, typer, toml, roman, tqdm).

Strategy: a curated dictionary of common informal romanized Nepali (and a few English) words for
high accuracy where it matters, with a rule-based OPTITRANS fallback for everything else. Devanagari,
punctuation, and digits pass through unchanged. Informal romanization has no fixed spelling, so the app
shows an EDITABLE Devanagari preview as the safety net. Extend `_WORD_MAP` to buy back accuracy over time.
"""
import re
from functools import lru_cache

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# High-frequency informal romanized Nepali (+ a few English) words -> correct Devanagari.
_WORD_MAP = {
    # pronouns / function words
    "ma": "म", "malai": "मलाई", "mero": "मेरो", "hami": "हामी", "hamro": "हाम्रो",
    "timi": "तिमी", "timro": "तिम्रो", "tapai": "तपाईं", "tapailai": "तपाईंलाई", "tapaiko": "तपाईंको",
    "yo": "यो", "tyo": "त्यो", "euta": "एउटा", "ani": "अनि", "tara": "तर", "ra": "र",
    # verbs / copula
    "ho": "हो", "hoina": "होइन", "cha": "छ", "chha": "छ", "chaina": "छैन", "chhaina": "छैन",
    "huncha": "हुन्छ", "hunuhuncha": "हुनुहुन्छ", "garcha": "गर्छ", "garchu": "गर्छु", "bhayo": "भयो",
    # question words / time
    "kasto": "कस्तो", "kasari": "कसरी", "kina": "किन", "kahile": "कहिले", "kaha": "कहाँ", "ke": "के",
    "aaile": "अहिले", "ahile": "अहिले", "aaja": "आज", "bholi": "भोलि", "hijo": "हिजो",
    # greetings / common nouns
    "namaste": "नमस्ते", "namaskar": "नमस्कार", "sanchai": "सञ्चै", "thik": "ठीक", "ramro": "राम्रो",
    "naam": "नाम", "nam": "नाम", "khana": "खाना", "pani": "पानी", "khabar": "खबर", "halkhabar": "हालखबर",
    "mausam": "मौसम", "manche": "मान्छे", "kathmandu": "काठमाडौं", "nepal": "नेपाल", "nepali": "नेपाली",
    "dhanyabad": "धन्यवाद", "maya": "माया", "sathi": "साथी", "saathi": "साथी",
    # common English -> nativized Devanagari (v1: single-script so espeak-ne stays in one language)
    "hello": "हेलो", "hi": "हाई", "ok": "ओके", "okay": "ओके", "thanks": "थ्याङ्क्स",
    "sorry": "सरी", "please": "प्लिज", "yes": "यस", "no": "नो",
}

# Informal spelling -> OPTITRANS conventions for the rule fallback (longer rules first).
_NORM = [
    ("chh", "Ch"), ("aa", "A"), ("ee", "I"), ("ii", "I"), ("oo", "U"), ("uu", "U"),
    ("w", "v"), ("z", "j"), ("x", "kS"),
]


def _normalize(tok: str) -> str:
    t = tok.lower()
    for a, b in _NORM:
        t = t.replace(a, b)
    return t


@lru_cache(maxsize=8192)
def _word(tok: str) -> str:
    low = tok.lower()
    if low in _WORD_MAP:
        return _WORD_MAP[low]
    return transliterate(_normalize(tok), sanscript.OPTITRANS, sanscript.DEVANAGARI)


def to_devanagari(text: str) -> str:
    """Romanized / code-mixed Latin Nepali -> Devanagari. Devanagari / punctuation / digits pass through."""
    if not text:
        return ""
    out = []
    for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", text):
        out.append(_word(tok) if (tok.isascii() and tok.isalpha()) else tok)
    return "".join(out)


# Documented code-mixing entry point (v1: single-script normalization to Devanagari).
handle_mixed = to_devanagari


if __name__ == "__main__":
    for s in [
        "kasto cha tapaiko aaile",
        "hello sanchai hunuhuncha tapai",
        "namaste, tapailai kasto cha?",
        "mero naam suryansh ho",
        "नमस्ते (already Devanagari) 2026",
    ]:
        print(f"{s}\n  -> {to_devanagari(s)}\n")
