"""Text normalisation and tokenisation that works for mixed Thai/English text.

Thai government PDFs have two recurring extraction problems this module fixes:
  1. SARA AM split: many PDFs encode "ำ" as NIKHAHIT + SARA AA ("ํา"), so "กำหนด"
     extracts as "กําหนด" and never matches a query typed on a normal keyboard.
  2. Thai digits: limits are written "๒๐ มิลลิกรัมต่อลิตร"; users type "20".
Thai has no spaces between words, so Thai runs are word-segmented with PyThaiNLP (newmm).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

THAI_RE = re.compile(r"[฀-๿]")
_THAI_RUN = re.compile(r"[฀-๿]+")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SARA_AM_TONE = re.compile("ํ([่-๋])า")
_SARA_AM = re.compile("ํ ?า")
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

EN_STOP = frozenset(
    """a an and are as at be by for from has have if in into is it its of on or that the
    their there these this to was were what when where which who will with does do how can
    i my me we our you your under must should""".split()
)


def normalize(text: str) -> str:
    """Canonical form used for both indexing and display-independent matching."""
    text = unicodedata.normalize("NFC", text)
    # ํ + (tone mark) + า  ->  (tone mark) + ำ   e.g. "นํ้า" -> "น้ำ"
    text = _SARA_AM_TONE.sub("\\1\u0E33", text)
    text = _SARA_AM.sub("\u0E33", text)             # ํ + า -> ำ  (also with a stray layout space)
    text = text.translate(_THAI_DIGITS)
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def is_thai(text: str, threshold: float = 0.3) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if THAI_RE.match(c)) / len(letters) >= threshold


# Transliterated technical terms that the default newmm dictionary shatters into meaningless
# syllables ("บีโอดี" -> "บี" + "โอดี"). Only add atomic terms here: adding a compound such as
# "ของเสียอันตราย" would stop a query for "ของเสีย" from matching it.
DOMAIN_TERMS = frozenset("""
บีโอดี ซีโอดี ทีเคเอ็น ทีดีเอส พีเอช ฟอร์มาลดีไฮด์ ซัลไฟด์ ไซยาไนด์ ฟีนอล แคดเมียม โครเมียม
ราชกิจจานุเบกษา
""".split())


@lru_cache(maxsize=1)
def _thai_segmenter():
    # imported lazily: PyThaiNLP is slow to load
    from pythainlp.corpus import thai_words
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import dict_trie

    trie = dict_trie(set(thai_words()) | DOMAIN_TERMS)
    return lambda text: word_tokenize(text, custom_dict=trie, engine="newmm", keep_whitespace=False)


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: Thai words via newmm, other scripts via a simple word regex."""
    text = normalize(text).lower()
    tokens: list[str] = []
    pos = 0
    for m in _THAI_RUN.finditer(text):
        tokens.extend(_latin_tokens(text[pos:m.start()]))
        seg = _thai_segmenter()(m.group())
        tokens.extend(t for t in seg if t.strip())
        pos = m.end()
    tokens.extend(_latin_tokens(text[pos:]))
    return tokens


def _latin_tokens(s: str) -> list[str]:
    return [t for t in _WORD.findall(s) if t not in EN_STOP]
