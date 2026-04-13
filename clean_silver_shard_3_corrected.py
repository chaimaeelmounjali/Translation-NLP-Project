#!/usr/bin/env python3
"""
Advanced multilingual corpus cleaning for MariNova.

This script upgrades a multilingual CSV from silver quality to a strict,
production-oriented gold-ready format.

Input columns expected exactly:
[
    'data_id',
    'id',
    'classe',
    'darija_arabic',
    'darija_arabizi',
    'english',
    'modern_standard_arabic',
    'english_word_count',
    'status',
    'qc_changed_fields',
    'qc_notes',
]

Outputs:
1) Cleaned CSV with status values in {CLEAN, FLAGGED, REMOVED}
2) JSON report with processing summary and issue counts
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm


EXPECTED_COLUMNS = [
    "data_id",
    "id",
    "classe",
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
    "english_word_count",
    "status",
    "qc_changed_fields",
    "qc_notes",
]

TEXT_COLUMNS = [
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
]

STATUS_CLEAN = "CLEAN"
STATUS_FLAGGED = "FLAGGED"
STATUS_REMOVED = "REMOVED"


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
ARTIFACT_PATTERN = re.compile(r"(?i)(?:\[unk\]|<unk>|@-@|@=@)")
MULTISPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?،؛؟])")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#[A-Za-z0-9_\u0600-\u06FF]+")
INVISIBLE_CHAR_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060]")
MOJIBAKE_HINT_PATTERN = re.compile(r"(?:Ã.|Â.|â.|Ø.|Ù.)")

ENGLISH_REPEAT_PATTERN = re.compile(r"([a-z])\1{2,}")
ARABIZI_REPEAT_PATTERN = re.compile(r"([a-z])\1{2,}")
ARABIC_REPEAT_PATTERN = re.compile(r"([\u0621-\u064A\u0671\u06A4\u06AD\u06AF\u06BA\u06C0\u06D2])\1{2,}")
ENGLISH_ABBREVIATION_PATTERN = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b")
ARABIC_LETTER_PATTERN = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTER_PATTERN = re.compile(r"[A-Za-z]")
ARABIC_LAUGH_PATTERN = re.compile(r"(?:ه){3,}")
LATIN_LAUGH_PATTERN = re.compile(r"(?:ha){2,}|h{3,}", flags=re.IGNORECASE)
ARABIC_WORD_PATTERN = re.compile(r"\b[\u0621-\u064A]{2,}\b")

ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_TATWEEL_PATTERN = re.compile(r"\u0640+")
ARABIC_ALEF_VARIANTS_PATTERN = re.compile(r"[\u0622\u0623\u0625\u0671]")
ARABIC_END_TA_MARBUTA_PATTERN = re.compile(r"ة\b")
ARABIC_END_ALIF_MAKSURA_PATTERN = re.compile(r"ى\b")


QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u0060": "'",
        "\u00B4": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u00AB": '"',
        "\u00BB": '"',
    }
)


MOJIBAKE_REPLACEMENTS = {
    "\ufeff": " ",
    "\u200b": " ",
    "\u200c": " ",
    "\u200d": " ",
    "\ufffd": " ",
}


PERSIAN_URDU_TO_ARABIC_TABLE = str.maketrans(
    {
        "ک": "ك",
        "ی": "ي",
        "ے": "ي",
        "ۍ": "ي",
        "ێ": "ي",
        "ھ": "ه",
        "ہ": "ه",
        "ۀ": "ه",
        "ؤ": "ؤ",
        "ئ": "ئ",
        "پ": "ب",
        "چ": "ج",
        "ژ": "ز",
    }
)


ENGLISH_CONTRACTION_FIXES = {
    r"\bdont\b": "don't",
    r"\bcant\b": "can't",
    r"\bwont\b": "won't",
    r"\bisnt\b": "isn't",
    r"\baren't\b": "aren't",
    r"\barent\b": "aren't",
    r"\bdoesnt\b": "doesn't",
    r"\bdidnt\b": "didn't",
    r"\bshouldnt\b": "shouldn't",
    r"\bcouldnt\b": "couldn't",
    r"\bwouldnt\b": "wouldn't",
    r"\bim\b": "i'm",
    r"\bive\b": "i've",
    r"\bill\b": "i'll",
    r"\byoure\b": "you're",
    r"\btheyre\b": "they're",
    r"\bweve\b": "we've",
    r"\bthats\b": "that's",
}


ENGLISH_CONTRACTION_EXPANSIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
}


COMMON_ENGLISH_TYPO_FIXES = {
    "teh": "the",
    "definitly": "definitely",
    "recieve": "receive",
    "adress": "address",
    "becuase": "because",
    "seperate": "separate",
    "wierd": "weird",
    "langauge": "language",
    "transaltion": "translation",
    "goverment": "government",
}


ARABIZI_VARIANT_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"\bd4\b"), "d"),
    (re.compile(r"(?<=\w)0(?=\w)"), "o"),
    (re.compile(r"\bwahed\b"), "wa7d"),
    (re.compile(r"\bwahd\b"), "wa7d"),
    (re.compile(r"\bel\b"), "l"),
)


@dataclass(frozen=True)
class LengthCheckConfig:
    """Thresholds to detect suspiciously imbalanced multilingual rows."""

    char_ratio_threshold: float = 8.0
    word_ratio_threshold: float = 6.0
    min_chars_for_ratio: int = 3


@dataclass(frozen=True)
class StopwordConfig:
    """Toggle stopword removal per language family."""

    remove_english: bool = True
    remove_arabic: bool = True
    remove_arabizi: bool = True


@dataclass(frozen=True)
class AdvancedNormalizationConfig:
    """Advanced normalization rules configurable from CLI."""

    arabic_final_taa_mode: str = "haa"  # haa | taa | keep
    darija_g_standard: str = "ڭ"        # ڭ | گ | غ
    darija_v_standard: str = "ف"        # ف | ڤ
    arabizi_kh_standard: str = "kh"     # kh | 5 | keep
    laugh_mode: str = "token"           # token | reduce | keep
    laugh_token: str = "<LAUGH>"
    normalize_hamza: bool = True
    normalize_arabizi_prefixes: bool = True
    normalize_darija_prefixes: bool = True
    expand_english_contractions: bool = True
    apply_basic_english_spell_fixes: bool = True
    min_words_per_field: int = 2
    max_words_per_field: int = 150
    missing_value_policy: str = "flag"  # flag | remove


DEFAULT_ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "that",
    "the", "their", "theirs", "them", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your", "yours",
}


DEFAULT_ARABIC_STOPWORDS: Set[str] = {
    "في", "من", "الى", "إلى", "على", "عن", "ما", "لا", "لم", "لن", "هو", "هي",
    "هم", "هذا", "هذه", "ذلك", "تلك", "هناك", "ثم", "او", "أو", "اذا", "إذا", "كان",
    "كانت", "يكون", "قد", "لقد", "ديال", "هاد", "داك", "حيت", "غادي", "واش", "بزاف",
}


DEFAULT_ARABIZI_STOPWORDS: Set[str] = {
    "f", "fi", "mn", "men", "3la", "w", "ma", "la", "ila", "l", "b", "dyal",
    "dial", "had", "dak", "hadi", "hada", "ghadi", "bzaf", "wach", "7it",
}


def parse_stopword_argument(raw_value: str | None) -> Set[str]:
    """Parse comma-separated stopwords passed from CLI."""
    if not raw_value:
        return set()
    return {tok.strip() for tok in raw_value.split(",") if tok.strip()}


def load_stopwords_from_file(path_value: str | None) -> Set[str]:
    """Load one stopword per line from a file, ignoring blank lines/comments."""
    if not path_value:
        return set()

    path = Path(path_value)
    if not path.exists():
        logging.warning("Stopword file not found: %s", path)
        return set()

    loaded: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        loaded.add(stripped)
    return loaded


def normalize_english_stopword(token: str) -> str:
    """Normalize stopword token with the same steps used for English text."""
    default_cfg = AdvancedNormalizationConfig()
    normalized = global_clean_text(token).lower()
    normalized = enforce_latin_punctuation(normalized)
    normalized = normalize_english_abbreviations(normalized)
    normalized = fix_common_english_contractions(normalized)
    if default_cfg.expand_english_contractions:
        normalized = expand_english_contractions(normalized)
    if default_cfg.apply_basic_english_spell_fixes:
        normalized = apply_basic_english_spelling_fixes(normalized)
    normalized = ENGLISH_REPEAT_PATTERN.sub(r"\1", normalized)
    return normalize_whitespace(normalized)


def normalize_arabizi_stopword(token: str) -> str:
    """Normalize stopword token with the same steps used for Arabizi text."""
    default_cfg = AdvancedNormalizationConfig()
    normalized = global_clean_text(token).lower()
    normalized = enforce_latin_punctuation(normalized)
    normalized = normalize_arabizi_variants(normalized, advanced_config=default_cfg)
    normalized = ARABIZI_REPEAT_PATTERN.sub(r"\1", normalized)
    return normalize_whitespace(normalized)


def normalize_arabic_stopword(token: str) -> str:
    """Normalize stopword token with the same steps used for Arabic script text."""
    default_cfg = AdvancedNormalizationConfig()
    normalized = global_clean_text(token)
    normalized = normalize_arabic_script(
        normalized,
        advanced_config=default_cfg,
        is_darija=False,
    )
    return normalize_whitespace(normalized)


def build_stopword_set(base: Set[str], extra_inline: str | None, extra_file: str | None, normalizer) -> Set[str]:
    """Build final normalized stopword set from defaults + CLI additions."""
    raw_values = set(base)
    raw_values.update(parse_stopword_argument(extra_inline))
    raw_values.update(load_stopwords_from_file(extra_file))

    normalized: Set[str] = set()
    for token in raw_values:
        cleaned = normalizer(token)
        if cleaned:
            # Keep multi-token entries by splitting after normalization.
            normalized.update(part for part in cleaned.split() if part)
    return normalized


def remove_stopwords(text: str, stopwords: Set[str]) -> str:
    """Drop stopwords token-by-token from already-normalized text."""
    if not text or not stopwords:
        return text
    kept = [tok for tok in text.split() if tok not in stopwords]
    return " ".join(kept)


def setup_logging(level: str) -> None:
    """Configure project-level logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_text(value: object) -> str:
    """Return empty string for missing values; cast any other value to string."""
    if pd.isna(value):
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and remove extra spaces before punctuation."""
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    return text


def normalize_quotes_and_apostrophes(text: str) -> str:
    """Unify quote and apostrophe variants into standard ASCII forms."""
    return text.translate(QUOTE_TRANSLATION_TABLE)


def strip_latin_accents(text: str) -> str:
    """Remove combining accents (useful for noisy Arabizi/French keyboard input)."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def remove_encoding_noise(text: str) -> str:
    """Remove common replacement and zero-width characters from encoding issues."""
    cleaned = text
    for bad, replacement in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, replacement)
    cleaned = CONTROL_CHAR_PATTERN.sub(" ", cleaned)
    cleaned = INVISIBLE_CHAR_PATTERN.sub(" ", cleaned)
    return cleaned


def maybe_fix_mojibake(text: str) -> str:
    """Try repairing common UTF-8/latin1 mojibake when hints are detected."""
    if not text or not MOJIBAKE_HINT_PATTERN.search(text):
        return text

    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text

    if not repaired:
        return text

    old_hits = len(MOJIBAKE_HINT_PATTERN.findall(text))
    new_hits = len(MOJIBAKE_HINT_PATTERN.findall(repaired))
    if new_hits < old_hits:
        return repaired
    return text


def remove_web_noise(text: str) -> str:
    """Remove URLs, emails, @mentions and #hashtags."""
    cleaned = URL_PATTERN.sub(" ", text)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(" ", cleaned)
    return cleaned


def normalize_laughter(text: str, is_arabic_script: bool, mode: str, laugh_token: str) -> str:
    """Normalize laugh patterns either to token, reduced form, or keep unchanged."""
    if mode == "keep":
        return text

    pattern = ARABIC_LAUGH_PATTERN if is_arabic_script else LATIN_LAUGH_PATTERN
    if mode == "token":
        return pattern.sub(f" {laugh_token} ", text)
    if mode == "reduce":
        replacement = "هه" if is_arabic_script else "hh"
        return pattern.sub(replacement, text)
    return text


def normalize_english_abbreviations(text: str) -> str:
    """Normalize dotted abbreviations like U.S.A. -> USA."""

    def _strip_dots(match: re.Match[str]) -> str:
        return match.group(0).replace(".", "")

    return ENGLISH_ABBREVIATION_PATTERN.sub(_strip_dots, text)


def apply_basic_english_spelling_fixes(text: str) -> str:
    """Apply basic typo corrections for common English misspellings."""
    fixed_tokens: List[str] = []
    for token in text.split():
        fixed_tokens.append(COMMON_ENGLISH_TYPO_FIXES.get(token, token))
    return " ".join(fixed_tokens)


def expand_english_contractions(text: str) -> str:
    """Expand contractions to full forms (e.g., won't -> will not)."""
    expanded = text
    for src, dst in sorted(ENGLISH_CONTRACTION_EXPANSIONS.items(), key=lambda kv: -len(kv[0])):
        expanded = re.sub(rf"\b{re.escape(src)}\b", dst, expanded)
    return expanded


def enforce_latin_punctuation(text: str) -> str:
    """Convert Arabic punctuation to latin punctuation for latin-script columns."""
    return text.replace("،", ",").replace("؛", ";").replace("؟", "?")


def normalize_hamza_forms(text: str) -> str:
    """Apply strict hamza normalization to reduce common variant spellings."""
    normalized = text.replace("ؤ", "و").replace("ئ", "ي")
    normalized = normalized.replace("ء", "")
    return normalized


def normalize_arabic_letter_variants(text: str) -> str:
    """Normalize Persian/Urdu keyboard letters to Arabic baseline letters."""
    return text.translate(PERSIAN_URDU_TO_ARABIC_TABLE)


def normalize_darija_specific_letters(text: str, g_standard: str, v_standard: str) -> str:
    """Unify Darija-specific grapheme choices for G and V families."""
    normalized = text
    if g_standard in {"ڭ", "گ", "غ"}:
        normalized = normalized.replace("ڭ", g_standard).replace("گ", g_standard).replace("ڭ", g_standard)
        # Requested by project checklist: optionally fold ghain into chosen G standard.
        normalized = normalized.replace("غ", g_standard)

    if v_standard == "ف":
        normalized = normalized.replace("ڤ", "ف")
    elif v_standard == "ڤ":
        normalized = normalized.replace("ڥ", "ڤ")

    return normalized


def normalize_darija_prefix_attachment_arabic(text: str) -> str:
    """Normalize spacing around attached one-letter Arabic prefixes (w/f/b/l)."""
    return re.sub(r"\b([وفبل])\s+(ال[\u0621-\u064A]+)", r"\1\2", text)


def normalize_arabizi_prefix_spacing(text: str) -> str:
    """Normalize Arabizi prefix forms like f'ddar -> f ddar and l'mdina -> l mdina."""
    normalized = re.sub(r"\b([wfbl])['’]([a-z0-9]{2,})\b", r"\1 \2", text)
    normalized = normalized.replace("'", "")
    return normalized


def normalize_arabizi_kh_standard(text: str, kh_standard: str) -> str:
    """Standardize kh sound notation between 'kh' and '5'."""
    if kh_standard == "keep":
        return text
    if kh_standard == "kh":
        # Keep numbers in other contexts; only replace likely letter-5 usage.
        text = re.sub(r"\b5(?=[a-z])", "kh", text)
        text = re.sub(r"(?<=[a-z])5\b", "kh", text)
        text = re.sub(r"(?<=[a-z])5(?=[a-z])", "kh", text)
        return text
    if kh_standard == "5":
        return text.replace("kh", "5")
    return text


def reduce_arabic_letter_repeats(text: str) -> str:
    """Reduce exaggerated Arabic elongations (e.g., بزااااف -> بزاف)."""
    return ARABIC_REPEAT_PATTERN.sub(r"\1", text)


def detect_script_mismatch_issues(cleaned_texts: Dict[str, str]) -> List[str]:
    """Heuristic script/lang mismatch checks across target columns."""
    issues: List[str] = []

    english = cleaned_texts.get("english", "")
    arabizi = cleaned_texts.get("darija_arabizi", "")
    darija_ar = cleaned_texts.get("darija_arabic", "")
    msa = cleaned_texts.get("modern_standard_arabic", "")

    if ARABIC_LETTER_PATTERN.search(english):
        issues.append("flagged_langid_english_contains_arabic")
    if ARABIC_LETTER_PATTERN.search(arabizi):
        issues.append("flagged_langid_arabizi_contains_arabic")
    if LATIN_LETTER_PATTERN.search(darija_ar):
        issues.append("flagged_langid_darija_arabic_contains_latin")
    if LATIN_LETTER_PATTERN.search(msa):
        issues.append("flagged_langid_msa_contains_latin")

    return issues


def transliterate_arabizi_for_alignment(text: str) -> str:
    """Rough transliteration for alignment checks only (not for final output text)."""
    normalized = text.lower()
    normalized = normalized.replace("kh", "خ").replace("gh", "غ").replace("sh", "ش").replace("ch", "ش")
    normalized = normalized.replace("th", "ث").replace("dh", "ذ")

    char_map = {
        "2": "ء",
        "3": "ع",
        "5": "خ",
        "6": "ط",
        "7": "ح",
        "8": "غ",
        "9": "ق",
        "a": "ا",
        "b": "ب",
        "d": "د",
        "f": "ف",
        "g": "ج",
        "h": "ه",
        "i": "ي",
        "j": "ج",
        "k": "ك",
        "l": "ل",
        "m": "م",
        "n": "ن",
        "o": "و",
        "q": "ق",
        "r": "ر",
        "s": "س",
        "t": "ت",
        "u": "و",
        "w": "و",
        "y": "ي",
        "z": "ز",
    }

    out_chars: List[str] = []
    for ch in normalized:
        out_chars.append(char_map.get(ch, ch))
    out = "".join(out_chars)
    out = re.sub(r"[^\u0621-\u064A\s]", " ", out)
    return normalize_whitespace(out)


def detect_cross_darija_arabizi_incoherence(cleaned_texts: Dict[str, str]) -> bool:
    """Flag when Darija Arabic and Arabizi appear semantically disconnected."""
    darija_ar = cleaned_texts.get("darija_arabic", "")
    arabizi = cleaned_texts.get("darija_arabizi", "")

    if len(darija_ar) < 12 or len(arabizi) < 12:
        return False

    darija_tokens = set(ARABIC_WORD_PATTERN.findall(darija_ar))
    arabizi_as_ar = transliterate_arabizi_for_alignment(arabizi)
    arabizi_tokens = set(ARABIC_WORD_PATTERN.findall(arabizi_as_ar))

    # Need enough lexical material to judge coherence.
    if len(darija_tokens) < 3 or len(arabizi_tokens) < 3:
        return False

    overlap = darija_tokens.intersection(arabizi_tokens)
    overlap_ratio = len(overlap) / max(1, min(len(darija_tokens), len(arabizi_tokens)))

    return overlap_ratio < 0.08


def within_word_bounds(cleaned_texts: Dict[str, str], min_words: int, max_words: int) -> Tuple[bool, str]:
    """Check whether each non-empty target field is within configured word bounds."""
    violations: List[str] = []
    for col, text in cleaned_texts.items():
        if not text:
            continue
        wc = len(text.split())
        if wc < min_words:
            violations.append(f"{col}:too_short({wc})")
        elif wc > max_words:
            violations.append(f"{col}:too_long({wc})")

    if violations:
        return False, "; ".join(violations)
    return True, ""


def make_loose_signature(cleaned_texts: Dict[str, str]) -> Tuple[str, str, str, str]:
    """Create punctuation-agnostic signatures to catch near-duplicate rows."""

    def _normalize(text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"[^\w\u0600-\u06FF]+", " ", lowered)
        lowered = normalize_whitespace(lowered)
        return lowered

    return (
        _normalize(cleaned_texts["darija_arabic"]),
        _normalize(cleaned_texts["darija_arabizi"]),
        _normalize(cleaned_texts["english"]),
        _normalize(cleaned_texts["modern_standard_arabic"]),
    )


def global_clean_text(value: object) -> str:
    """
    Global cleaning applied to every text field.

    Includes:
    - HTML/XML stripping
    - escaped/newline/tab cleanup
    - artifact removal
    - quote normalization
    - spacing normalization
    """
    text = safe_text(value)
    text = maybe_fix_mojibake(text)
    text = html.unescape(text)
    text = remove_encoding_noise(text)
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = remove_web_noise(text)
    text = ARTIFACT_PATTERN.sub(" ", text)
    text = normalize_quotes_and_apostrophes(text)
    return normalize_whitespace(text)


def fix_common_english_contractions(text: str) -> str:
    """Repair frequent missing-apostrophe contractions."""
    fixed = text
    for pattern, replacement in ENGLISH_CONTRACTION_FIXES.items():
        fixed = re.sub(pattern, replacement, fixed)
    return fixed


def clean_english(
    value: object,
    english_stopwords: Set[str],
    remove_sw: bool,
    advanced_config: AdvancedNormalizationConfig,
) -> str:
    """Language-specific English cleaning rules."""
    text = global_clean_text(value).lower()
    text = enforce_latin_punctuation(text)
    text = normalize_english_abbreviations(text)
    text = fix_common_english_contractions(text)
    if advanced_config.expand_english_contractions:
        text = expand_english_contractions(text)
    if advanced_config.apply_basic_english_spell_fixes:
        text = apply_basic_english_spelling_fixes(text)
    text = normalize_laughter(
        text,
        is_arabic_script=False,
        mode=advanced_config.laugh_mode,
        laugh_token=advanced_config.laugh_token,
    )
    text = ENGLISH_REPEAT_PATTERN.sub(r"\1", text)
    text = normalize_whitespace(text)
    if remove_sw:
        text = remove_stopwords(text, english_stopwords)
    return normalize_whitespace(text)


def normalize_arabizi_variants(text: str, advanced_config: AdvancedNormalizationConfig) -> str:
    """Apply conservative Arabizi canonicalization for common noisy variants."""
    normalized = strip_latin_accents(text)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    for pattern, replacement in ARABIZI_VARIANT_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    if advanced_config.normalize_arabizi_prefixes:
        normalized = normalize_arabizi_prefix_spacing(normalized)
    normalized = normalize_arabizi_kh_standard(normalized, advanced_config.arabizi_kh_standard)
    return normalized


def clean_darija_arabizi(
    value: object,
    arabizi_stopwords: Set[str],
    remove_sw: bool,
    advanced_config: AdvancedNormalizationConfig,
) -> str:
    """Language-specific Darija Arabizi cleaning rules."""
    text = global_clean_text(value).lower()
    text = enforce_latin_punctuation(text)
    text = normalize_arabizi_variants(text, advanced_config=advanced_config)
    text = normalize_laughter(
        text,
        is_arabic_script=False,
        mode=advanced_config.laugh_mode,
        laugh_token=advanced_config.laugh_token,
    )
    text = ARABIZI_REPEAT_PATTERN.sub(r"\1", text)
    text = normalize_whitespace(text)
    if remove_sw:
        text = remove_stopwords(text, arabizi_stopwords)
    return normalize_whitespace(text)


def enforce_arabic_punctuation(text: str) -> str:
    """Convert ASCII punctuation to Arabic punctuation for Arabic-script fields."""
    text = text.replace(",", "،")
    text = text.replace(";", "؛")
    text = text.replace("?", "؟")
    return text


def normalize_arabic_script(
    text: str,
    advanced_config: AdvancedNormalizationConfig,
    is_darija: bool,
) -> str:
    """
    Arabic normalization shared by Darija Arabic and MSA.

    Rules:
    - Alef variants -> bare Alef
    - remove diacritics
    - remove tatweel
    - unify terminal Taa Marbouta/Haa (here: ة -> ه)
    - unify terminal Yaa/Alif Maqsoura (here: ى -> ي)
    """
    text = normalize_arabic_letter_variants(text)
    if is_darija:
        text = normalize_darija_specific_letters(
            text,
            g_standard=advanced_config.darija_g_standard,
            v_standard=advanced_config.darija_v_standard,
        )

    text = ARABIC_ALEF_VARIANTS_PATTERN.sub("ا", text)
    text = ARABIC_DIACRITICS_PATTERN.sub("", text)
    text = ARABIC_TATWEEL_PATTERN.sub("", text)
    if advanced_config.normalize_hamza:
        text = normalize_hamza_forms(text)

    if advanced_config.arabic_final_taa_mode == "haa":
        text = ARABIC_END_TA_MARBUTA_PATTERN.sub("ه", text)
    elif advanced_config.arabic_final_taa_mode == "taa":
        # Keep taa marbouta as-is and only convert likely keyboard-final haa to taa.
        text = re.sub(r"\b([\u0621-\u064A]{3,})ه\b", r"\1ة", text)

    text = ARABIC_END_ALIF_MAKSURA_PATTERN.sub("ي", text)
    if is_darija:
        text = reduce_arabic_letter_repeats(text)
        if advanced_config.normalize_darija_prefixes:
            text = normalize_darija_prefix_attachment_arabic(text)
    text = normalize_laughter(
        text,
        is_arabic_script=True,
        mode=advanced_config.laugh_mode,
        laugh_token=advanced_config.laugh_token,
    )
    text = enforce_arabic_punctuation(text)
    return normalize_whitespace(text)


def clean_darija_arabic(
    value: object,
    arabic_stopwords: Set[str],
    remove_sw: bool,
    advanced_config: AdvancedNormalizationConfig,
) -> str:
    """Language-specific Darija Arabic cleaning rules."""
    text = global_clean_text(value)
    text = normalize_arabic_script(text, advanced_config=advanced_config, is_darija=True)
    if remove_sw:
        text = remove_stopwords(text, arabic_stopwords)
    return normalize_whitespace(text)


def clean_msa(
    value: object,
    arabic_stopwords: Set[str],
    remove_sw: bool,
    advanced_config: AdvancedNormalizationConfig,
) -> str:
    """Language-specific MSA cleaning rules."""
    text = global_clean_text(value)
    text = normalize_arabic_script(text, advanced_config=advanced_config, is_darija=False)
    if remove_sw:
        text = remove_stopwords(text, arabic_stopwords)
    return normalize_whitespace(text)


def detect_missing_values(raw_row: Dict[str, object], cleaned_texts: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Return columns with NaN values and columns empty after cleaning."""
    nan_columns: List[str] = []
    empty_columns: List[str] = []

    for col in TEXT_COLUMNS:
        if pd.isna(raw_row.get(col)):
            nan_columns.append(col)
        if cleaned_texts.get(col, "") == "":
            empty_columns.append(col)

    return nan_columns, empty_columns


def detect_extreme_length_discrepancy(
    cleaned_texts: Dict[str, str],
    config: LengthCheckConfig,
) -> Tuple[bool, str]:
    """
    Flag rows where multilingual lengths are suspiciously imbalanced.

    Uses both character-length ratio and token-length ratio.
    """
    char_lengths = {col: len(txt) for col, txt in cleaned_texts.items() if txt}
    word_lengths = {col: len(txt.split()) for col, txt in cleaned_texts.items() if txt}

    if len(char_lengths) < 2 or len(word_lengths) < 2:
        return False, ""

    char_candidates = {
        col: length
        for col, length in char_lengths.items()
        if length >= config.min_chars_for_ratio
    }
    if len(char_candidates) >= 2:
        min_char_col = min(char_candidates, key=char_candidates.get)
        max_char_col = max(char_candidates, key=char_candidates.get)
        min_char = char_candidates[min_char_col]
        max_char = char_candidates[max_char_col]
        char_ratio = max_char / max(1, min_char)
    else:
        min_char_col = max_char_col = ""
        min_char = max_char = 0
        char_ratio = 1.0

    min_word_col = min(word_lengths, key=word_lengths.get)
    max_word_col = max(word_lengths, key=word_lengths.get)
    min_word = word_lengths[min_word_col]
    max_word = word_lengths[max_word_col]
    word_ratio = max_word / max(1, min_word)

    reasons: List[str] = []
    if char_ratio >= config.char_ratio_threshold:
        reasons.append(
            f"char_ratio={char_ratio:.2f} ({min_char_col}:{min_char} vs {max_char_col}:{max_char})"
        )
    if word_ratio >= config.word_ratio_threshold:
        reasons.append(
            f"word_ratio={word_ratio:.2f} ({min_word_col}:{min_word} vs {max_word_col}:{max_word})"
        )

    if reasons:
        return True, " ; ".join(reasons)
    return False, ""


def build_qc_note(
    status: str,
    modified_fields: Sequence[str],
    nan_columns: Sequence[str],
    empty_columns: Sequence[str],
    issues: Sequence[str],
    discrepancy_detail: str,
) -> str:
    """Create compact human-readable QC notes in English."""
    notes: List[str] = []
    issue_set = set(issues)

    if modified_fields:
        notes.append("Normalized fields: " + ", ".join(modified_fields) + ".")

    if "removed_empty_row" in issue_set:
        notes.append("Row removed: all four text fields are empty after cleaning.")

    if "removed_duplicate_data_id" in issue_set:
        notes.append("Row removed: duplicate data_id detected.")

    if "removed_duplicate_multilingual_text" in issue_set:
        notes.append("Row removed: exact duplicate multilingual text detected.")

    if "removed_duplicate_partial_text" in issue_set:
        notes.append("Row removed: near-duplicate multilingual text detected (partial duplicate).")

    if "removed_nan_values" in issue_set:
        notes.append("Row removed because missing values were detected.")

    if "removed_word_count_out_of_bounds" in issue_set:
        notes.append("Row removed due to very short/very long sentence length.")

    if nan_columns:
        notes.append("NaN detected in: " + ", ".join(nan_columns) + ".")

    if empty_columns and "removed_empty_row" not in issue_set:
        notes.append("Empty after cleaning in: " + ", ".join(empty_columns) + ".")

    if "flagged_empty_after_cleaning" in issue_set:
        notes.append("One or more target fields became empty after cleaning.")

    if "flagged_length_discrepancy" in issue_set:
        note = "Extreme length discrepancy across languages"
        if discrepancy_detail:
            note += f" ({discrepancy_detail})"
        notes.append(note + ".")

    langid_issue_notes = {
        "flagged_langid_english_contains_arabic": "Language mismatch: English field contains Arabic script.",
        "flagged_langid_arabizi_contains_arabic": "Language mismatch: Arabizi field contains Arabic script.",
        "flagged_langid_darija_arabic_contains_latin": "Language mismatch: Darija Arabic field contains Latin script.",
        "flagged_langid_msa_contains_latin": "Language mismatch: MSA field contains Latin script.",
    }
    for issue_key, message in langid_issue_notes.items():
        if issue_key in issue_set:
            notes.append(message)

    if "flagged_cross_darija_arabizi_incoherence" in issue_set:
        notes.append("Cross-lingual mismatch suspected between Darija Arabic and Arabizi columns.")

    if status == STATUS_CLEAN and not notes:
        return ""

    return " ".join(notes)


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the input file contains the required dataset schema."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def process_dataframe(
    df: pd.DataFrame,
    length_config: LengthCheckConfig,
    english_stopwords: Set[str],
    arabic_stopwords: Set[str],
    arabizi_stopwords: Set[str],
    stopword_config: StopwordConfig,
    advanced_config: AdvancedNormalizationConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Clean, flag, and mark rows with robust per-row exception handling."""
    issue_counter: Counter[str] = Counter()
    modified_field_counter: Counter[str] = Counter()

    data_id_series = df["data_id"].fillna("").astype(str).str.strip()
    duplicate_data_id_indices = set(df.index[data_id_series.ne("") & data_id_series.duplicated(keep="first")])

    seen_multilingual_signatures: set[Tuple[str, str, str, str]] = set()
    seen_multilingual_loose_signatures: set[Tuple[str, str, str, str]] = set()
    cleaned_rows: List[Dict[str, object]] = []

    clean_count = 0
    removed_count = 0
    flagged_count = 0
    modified_row_count = 0

    iterator = tqdm(df.iterrows(), total=len(df), desc="Cleaning rows", unit="row")

    for row_index, row in iterator:
        try:
            row_dict = row.to_dict()

            original_texts = {col: safe_text(row_dict.get(col)) for col in TEXT_COLUMNS}
            normalized_original_for_diff = {
                col: normalize_whitespace(original_texts[col]) for col in TEXT_COLUMNS
            }

            cleaned_texts = {
                "darija_arabic": clean_darija_arabic(
                    row_dict.get("darija_arabic"),
                    arabic_stopwords=arabic_stopwords,
                    remove_sw=stopword_config.remove_arabic,
                    advanced_config=advanced_config,
                ),
                "darija_arabizi": clean_darija_arabizi(
                    row_dict.get("darija_arabizi"),
                    arabizi_stopwords=arabizi_stopwords,
                    remove_sw=stopword_config.remove_arabizi,
                    advanced_config=advanced_config,
                ),
                "english": clean_english(
                    row_dict.get("english"),
                    english_stopwords=english_stopwords,
                    remove_sw=stopword_config.remove_english,
                    advanced_config=advanced_config,
                ),
                "modern_standard_arabic": clean_msa(
                    row_dict.get("modern_standard_arabic"),
                    arabic_stopwords=arabic_stopwords,
                    remove_sw=stopword_config.remove_arabic,
                    advanced_config=advanced_config,
                ),
            }

            modified_fields = [
                col
                for col in TEXT_COLUMNS
                if cleaned_texts[col] != normalized_original_for_diff[col]
            ]
            if modified_fields:
                modified_row_count += 1
                for field_name in modified_fields:
                    modified_field_counter[field_name] += 1

            nan_columns, empty_columns = detect_missing_values(row_dict, cleaned_texts)
            row_issues: List[str] = []
            discrepancy_detail = ""

            all_empty = all(cleaned_texts[col] == "" for col in TEXT_COLUMNS)

            if all_empty:
                status = STATUS_REMOVED
                row_issues.append("removed_empty_row")
                issue_counter["removed_empty_row"] += 1
            elif row_index in duplicate_data_id_indices:
                status = STATUS_REMOVED
                row_issues.append("removed_duplicate_data_id")
                issue_counter["removed_duplicate_data_id"] += 1
            else:
                signature = (
                    cleaned_texts["darija_arabic"],
                    cleaned_texts["darija_arabizi"],
                    cleaned_texts["english"],
                    cleaned_texts["modern_standard_arabic"],
                )
                if signature in seen_multilingual_signatures:
                    status = STATUS_REMOVED
                    row_issues.append("removed_duplicate_multilingual_text")
                    issue_counter["removed_duplicate_multilingual_text"] += 1
                else:
                    seen_multilingual_signatures.add(signature)
                    loose_signature = make_loose_signature(cleaned_texts)
                    if loose_signature in seen_multilingual_loose_signatures:
                        status = STATUS_REMOVED
                        row_issues.append("removed_duplicate_partial_text")
                        issue_counter["removed_duplicate_partial_text"] += 1
                    else:
                        seen_multilingual_loose_signatures.add(loose_signature)
                        status = STATUS_CLEAN

            if status != STATUS_REMOVED:
                if nan_columns:
                    if advanced_config.missing_value_policy == "remove":
                        status = STATUS_REMOVED
                        row_issues.append("removed_nan_values")
                        issue_counter["removed_nan_values"] += 1
                    else:
                        status = STATUS_FLAGGED
                        row_issues.append("flagged_nan_values")
                        issue_counter["flagged_nan_values"] += 1

                partially_empty = [col for col in empty_columns if col not in nan_columns]
                if partially_empty:
                    status = STATUS_FLAGGED
                    row_issues.append("flagged_empty_after_cleaning")
                    issue_counter["flagged_empty_after_cleaning"] += 1

                lang_issues = detect_script_mismatch_issues(cleaned_texts)

                # Balanced profile: keep only hard language mismatches.
                if (
                    not advanced_config.normalize_hamza
                    and not advanced_config.normalize_arabizi_prefixes
                    and not advanced_config.normalize_darija_prefixes
                ):
                    allowed_lang_issues = {
                        "flagged_langid_english_contains_arabic",
                        "flagged_langid_arabizi_contains_arabic",
                    }
                    lang_issues = [issue for issue in lang_issues if issue in allowed_lang_issues]

                for issue in lang_issues:
                    status = STATUS_FLAGGED
                    row_issues.append(issue)
                    issue_counter[issue] += 1

                if (
                    advanced_config.normalize_arabizi_prefixes
                    and advanced_config.normalize_darija_prefixes
                    and detect_cross_darija_arabizi_incoherence(cleaned_texts)
                ):
                    status = STATUS_FLAGGED
                    row_issues.append("flagged_cross_darija_arabizi_incoherence")
                    issue_counter["flagged_cross_darija_arabizi_incoherence"] += 1

                in_bounds, bounds_detail = within_word_bounds(
                    cleaned_texts,
                    min_words=advanced_config.min_words_per_field,
                    max_words=advanced_config.max_words_per_field,
                )
                if not in_bounds:
                    status = STATUS_REMOVED
                    row_issues.append("removed_word_count_out_of_bounds")
                    issue_counter["removed_word_count_out_of_bounds"] += 1
                    if bounds_detail:
                        discrepancy_detail = (discrepancy_detail + " ; " + bounds_detail).strip(" ;")

                has_discrepancy, discrepancy_detail = detect_extreme_length_discrepancy(
                    cleaned_texts,
                    config=length_config,
                )
                if has_discrepancy and status != STATUS_REMOVED:
                    status = STATUS_FLAGGED
                    row_issues.append("flagged_length_discrepancy")
                    issue_counter["flagged_length_discrepancy"] += 1

            output_row = dict(row_dict)
            output_row.update(cleaned_texts)
            output_row["english_word_count"] = int(len(cleaned_texts["english"].split()))
            output_row["status"] = status

            changed_fields = set(modified_fields)
            if status in {STATUS_FLAGGED, STATUS_REMOVED}:
                changed_fields.add("status")
                changed_fields.update(nan_columns)
                changed_fields.update(empty_columns)

            output_row["qc_changed_fields"] = ";".join(sorted(changed_fields))
            output_row["qc_notes"] = build_qc_note(
                status=status,
                modified_fields=modified_fields,
                nan_columns=nan_columns,
                empty_columns=empty_columns,
                issues=row_issues,
                discrepancy_detail=discrepancy_detail,
            )

            if status == STATUS_CLEAN:
                clean_count += 1
            elif status == STATUS_FLAGGED:
                flagged_count += 1
            else:
                removed_count += 1

            cleaned_rows.append(output_row)

        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Row %s failed during processing: %s", row_index, exc)
            issue_counter["row_processing_exception"] += 1

            fallback_row = row.to_dict()
            for col in TEXT_COLUMNS:
                fallback_row[col] = global_clean_text(fallback_row.get(col))
            fallback_row["english_word_count"] = int(len(safe_text(fallback_row.get("english")).split()))
            fallback_row["status"] = STATUS_FLAGGED
            fallback_row["qc_changed_fields"] = "status"
            fallback_row["qc_notes"] = f"Row processing exception: {exc.__class__.__name__}."
            cleaned_rows.append(fallback_row)
            flagged_count += 1

    output_df = pd.DataFrame(cleaned_rows)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows_processed": int(len(df)),
        "removed_rows": int(removed_count),
        "cleaned_rows": int(clean_count),
        "flagged_rows": int(flagged_count),
        "modified_rows": int(modified_row_count),
        "issue_summary": dict(issue_counter),
        "column_modification_summary": dict(modified_field_counter),
        "length_discrepancy_thresholds": {
            "char_ratio_threshold": length_config.char_ratio_threshold,
            "word_ratio_threshold": length_config.word_ratio_threshold,
            "min_chars_for_ratio": length_config.min_chars_for_ratio,
        },
    }

    return output_df, report


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced modular multilingual text cleaner for MariNova corpus.",
    )
    parser.add_argument(
        "--input",
        default="artifacts/silver_shard_3_qc/silver_shard_3.corrected.auto.final.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/silver_shard_3_cleaned",
        help="Output directory for cleaned CSV and JSON report.",
    )
    parser.add_argument(
        "--output-name",
        default="silver_shard_3.corrected.cleaned.csv",
        help="Output cleaned CSV filename.",
    )
    parser.add_argument(
        "--report-name",
        default="cleaning_report.json",
        help="Output JSON report filename.",
    )
    parser.add_argument(
        "--drop-removed-rows",
        action="store_true",
        help="If set, rows marked REMOVED are excluded from the written CSV.",
    )
    parser.add_argument(
        "--char-ratio-threshold",
        type=float,
        default=8.0,
        help="Extreme char-length ratio threshold used for FLAGGED rows.",
    )
    parser.add_argument(
        "--word-ratio-threshold",
        type=float,
        default=6.0,
        help="Extreme word-length ratio threshold used for FLAGGED rows.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--disable-english-stopwords",
        action="store_true",
        help="Disable English stopword removal.",
    )
    parser.add_argument(
        "--disable-arabic-stopwords",
        action="store_true",
        help="Disable Arabic-script stopword removal for Darija Arabic and MSA.",
    )
    parser.add_argument(
        "--disable-arabizi-stopwords",
        action="store_true",
        help="Disable Darija Arabizi stopword removal.",
    )
    parser.add_argument(
        "--english-stopwords-add",
        default=None,
        help="Comma-separated extra English stopwords.",
    )
    parser.add_argument(
        "--arabic-stopwords-add",
        default=None,
        help="Comma-separated extra Arabic-script stopwords.",
    )
    parser.add_argument(
        "--arabizi-stopwords-add",
        default=None,
        help="Comma-separated extra Darija Arabizi stopwords.",
    )
    parser.add_argument(
        "--english-stopwords-file",
        default=None,
        help="Optional file with one English stopword per line.",
    )
    parser.add_argument(
        "--arabic-stopwords-file",
        default=None,
        help="Optional file with one Arabic-script stopword per line.",
    )
    parser.add_argument(
        "--arabizi-stopwords-file",
        default=None,
        help="Optional file with one Darija Arabizi stopword per line.",
    )
    parser.add_argument(
        "--arabic-final-taa-mode",
        default="haa",
        choices=["haa", "taa", "keep"],
        help="Arabic final-letter policy for ة/ه normalization.",
    )
    parser.add_argument(
        "--darija-g-standard",
        default="ڭ",
        choices=["ڭ", "گ", "غ"],
        help="Target standard for Darija G-family letters.",
    )
    parser.add_argument(
        "--darija-v-standard",
        default="ف",
        choices=["ف", "ڤ"],
        help="Target standard for Darija V-family letters.",
    )
    parser.add_argument(
        "--arabizi-kh-standard",
        default="kh",
        choices=["kh", "5", "keep"],
        help="Target standard for Arabizi kh sound.",
    )
    parser.add_argument(
        "--laugh-mode",
        default="token",
        choices=["token", "reduce", "keep"],
        help="How to normalize laugh patterns.",
    )
    parser.add_argument(
        "--laugh-token",
        default="<LAUGH>",
        help="Token used when --laugh-mode=token.",
    )
    parser.add_argument(
        "--disable-hamza-normalization",
        action="store_true",
        help="Disable strict hamza normalization.",
    )
    parser.add_argument(
        "--disable-darija-prefix-normalization",
        action="store_true",
        help="Disable Darija Arabic prefix attachment normalization.",
    )
    parser.add_argument(
        "--disable-arabizi-prefix-normalization",
        action="store_true",
        help="Disable Arabizi prefix spacing normalization.",
    )
    parser.add_argument(
        "--disable-english-contraction-expansion",
        action="store_true",
        help="Disable expansion of English contractions (won't -> will not).",
    )
    parser.add_argument(
        "--disable-english-typo-fixes",
        action="store_true",
        help="Disable basic English typo corrections.",
    )
    parser.add_argument(
        "--min-words-per-field",
        type=int,
        default=2,
        help="Minimum words per non-empty field before removal.",
    )
    parser.add_argument(
        "--max-words-per-field",
        type=int,
        default=150,
        help="Maximum words per field before removal.",
    )
    parser.add_argument(
        "--missing-value-policy",
        default="flag",
        choices=["flag", "remove"],
        help="Policy for rows containing NaN values in target text columns.",
    )
    parser.add_argument(
        "--cleaning-profile",
        default="strict",
        choices=["strict", "balanced"],
        help=(
            "strict: aggressive rules for maximal normalization/validation; "
            "balanced: lighter language/coherence checks for better retention."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_csv_path = output_dir / args.output_name
    report_path = output_dir / args.report_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading input CSV: %s", input_path)
    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=object)
    validate_columns(df)

    length_config = LengthCheckConfig(
        char_ratio_threshold=float(args.char_ratio_threshold),
        word_ratio_threshold=float(args.word_ratio_threshold),
        min_chars_for_ratio=3,
    )

    stopword_config = StopwordConfig(
        remove_english=not args.disable_english_stopwords,
        remove_arabic=not args.disable_arabic_stopwords,
        remove_arabizi=not args.disable_arabizi_stopwords,
    )

    advanced_config = AdvancedNormalizationConfig(
        arabic_final_taa_mode=args.arabic_final_taa_mode,
        darija_g_standard=args.darija_g_standard,
        darija_v_standard=args.darija_v_standard,
        arabizi_kh_standard=args.arabizi_kh_standard,
        laugh_mode=args.laugh_mode,
        laugh_token=args.laugh_token,
        normalize_hamza=not args.disable_hamza_normalization,
        normalize_arabizi_prefixes=not args.disable_arabizi_prefix_normalization,
        normalize_darija_prefixes=not args.disable_darija_prefix_normalization,
        expand_english_contractions=not args.disable_english_contraction_expansion,
        apply_basic_english_spell_fixes=not args.disable_english_typo_fixes,
        min_words_per_field=max(1, int(args.min_words_per_field)),
        max_words_per_field=max(1, int(args.max_words_per_field)),
        missing_value_policy=args.missing_value_policy,
    )

    if args.cleaning_profile == "balanced":
        # Keep core hygiene; relax rules that over-flag cross-script named entities.
        advanced_config = AdvancedNormalizationConfig(
            arabic_final_taa_mode=advanced_config.arabic_final_taa_mode,
            darija_g_standard=advanced_config.darija_g_standard,
            darija_v_standard=advanced_config.darija_v_standard,
            arabizi_kh_standard="keep",
            laugh_mode=advanced_config.laugh_mode,
            laugh_token=advanced_config.laugh_token,
            normalize_hamza=False,
            normalize_arabizi_prefixes=False,
            normalize_darija_prefixes=False,
            expand_english_contractions=advanced_config.expand_english_contractions,
            apply_basic_english_spell_fixes=advanced_config.apply_basic_english_spell_fixes,
            min_words_per_field=1,
            max_words_per_field=max(advanced_config.max_words_per_field, 180),
            missing_value_policy=advanced_config.missing_value_policy,
        )

    english_stopwords = build_stopword_set(
        base=DEFAULT_ENGLISH_STOPWORDS,
        extra_inline=args.english_stopwords_add,
        extra_file=args.english_stopwords_file,
        normalizer=normalize_english_stopword,
    )
    arabic_stopwords = build_stopword_set(
        base=DEFAULT_ARABIC_STOPWORDS,
        extra_inline=args.arabic_stopwords_add,
        extra_file=args.arabic_stopwords_file,
        normalizer=normalize_arabic_stopword,
    )
    arabizi_stopwords = build_stopword_set(
        base=DEFAULT_ARABIZI_STOPWORDS,
        extra_inline=args.arabizi_stopwords_add,
        extra_file=args.arabizi_stopwords_file,
        normalizer=normalize_arabizi_stopword,
    )

    logging.info(
        "Stopwords loaded | english=%s arabic=%s arabizi=%s | enabled=(%s,%s,%s)",
        len(english_stopwords),
        len(arabic_stopwords),
        len(arabizi_stopwords),
        stopword_config.remove_english,
        stopword_config.remove_arabic,
        stopword_config.remove_arabizi,
    )

    cleaned_df, report = process_dataframe(
        df,
        length_config=length_config,
        english_stopwords=english_stopwords,
        arabic_stopwords=arabic_stopwords,
        arabizi_stopwords=arabizi_stopwords,
        stopword_config=stopword_config,
        advanced_config=advanced_config,
    )

    if args.drop_removed_rows:
        before = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df["status"] != STATUS_REMOVED].copy()
        dropped = before - len(cleaned_df)
        logging.info("Dropped %s REMOVED rows from written CSV due to --drop-removed-rows", dropped)
        report["rows_dropped_from_output_because_removed"] = int(dropped)

    report["input_csv"] = str(input_path)
    report["output_csv"] = str(output_csv_path)
    report["report_json"] = str(report_path)
    report["rows_written_to_output_csv"] = int(len(cleaned_df))
    report["stopword_settings"] = {
        "english_enabled": stopword_config.remove_english,
        "arabic_enabled": stopword_config.remove_arabic,
        "arabizi_enabled": stopword_config.remove_arabizi,
        "english_count": len(english_stopwords),
        "arabic_count": len(arabic_stopwords),
        "arabizi_count": len(arabizi_stopwords),
        "english_add_arg": args.english_stopwords_add or "",
        "arabic_add_arg": args.arabic_stopwords_add or "",
        "arabizi_add_arg": args.arabizi_stopwords_add or "",
        "english_stopwords_file": args.english_stopwords_file or "",
        "arabic_stopwords_file": args.arabic_stopwords_file or "",
        "arabizi_stopwords_file": args.arabizi_stopwords_file or "",
    }
    report["advanced_normalization_settings"] = {
        "cleaning_profile": args.cleaning_profile,
        "arabic_final_taa_mode": advanced_config.arabic_final_taa_mode,
        "darija_g_standard": advanced_config.darija_g_standard,
        "darija_v_standard": advanced_config.darija_v_standard,
        "arabizi_kh_standard": advanced_config.arabizi_kh_standard,
        "laugh_mode": advanced_config.laugh_mode,
        "laugh_token": advanced_config.laugh_token,
        "normalize_hamza": advanced_config.normalize_hamza,
        "normalize_arabizi_prefixes": advanced_config.normalize_arabizi_prefixes,
        "normalize_darija_prefixes": advanced_config.normalize_darija_prefixes,
        "expand_english_contractions": advanced_config.expand_english_contractions,
        "apply_basic_english_spell_fixes": advanced_config.apply_basic_english_spell_fixes,
        "min_words_per_field": advanced_config.min_words_per_field,
        "max_words_per_field": advanced_config.max_words_per_field,
        "missing_value_policy": advanced_config.missing_value_policy,
    }

    # Preserve original column order when possible.
    output_columns = list(df.columns)
    for required in EXPECTED_COLUMNS:
        if required not in output_columns:
            output_columns.append(required)
    cleaned_df = cleaned_df.reindex(columns=output_columns)

    logging.info("Writing cleaned CSV: %s", output_csv_path)
    cleaned_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    logging.info("Writing JSON report: %s", report_path)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    logging.info(
        "Done. total=%s cleaned=%s flagged=%s removed=%s written=%s",
        report["total_rows_processed"],
        report["cleaned_rows"],
        report["flagged_rows"],
        report["removed_rows"],
        report["rows_written_to_output_csv"],
    )


if __name__ == "__main__":
    main()
